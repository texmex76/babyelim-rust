use bzip2;
use bzip2::read::BzDecoder;
use bzip2::write::BzEncoder;
use clap::{value_parser, Arg, ArgAction, Command};
use flate2;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use std::cell::RefCell;
use std::collections::LinkedList;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::ops::{Index, IndexMut};
use std::path::Path;
use std::process;
use std::rc::Rc;
use std::time::Instant;
use xz2::read::XzDecoder;
use xz2::write::XzEncoder;

macro_rules! die {
    ($($arg:tt)*) => {{
        eprintln!("babysub: error: {}", format!($($arg)*));
        process::exit(1);
    }}
}

macro_rules! message {
    ($verbosity:expr, $($arg:tt)*) => {{
        use std::io::{self, Write};
        if $verbosity >= 0 {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            if let Err(e) = writeln!(handle, "{}", format!("c {}", format_args!($($arg)*))) {
                die!("Failed to write message: {}", e);
            }
            if let Err(f) = handle.flush() {
                die!("Failed to flush stdout: {}", f);
            }
        }
    }}
}

macro_rules! verbose {
    ($verbosity:expr, $level:expr, $($arg:tt)*) => {{
        use std::io::{self, Write};
        if $verbosity >= $level {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            if let Err(e) = writeln!(handle, "{}", format!("c {}", format_args!($($arg)*))) {
                die!("Failed to write message: {}", e);
            }
            if let Err(f) = handle.flush() {
                die!("Failed to flush stdout: {}", f);
            }
        }
    }}
}

macro_rules! parse_error {
    ($ctx:expr, $msg:expr, $line:expr) => {{
        eprintln!(
            "babysub: parse error: at line {} in '{}': {}",
            $line, $ctx.config.input_path, $msg
        );
        process::exit(1);
    }};
}

#[cfg(feature = "logging")]
macro_rules! LOG {
    ($verbosity:expr, $($arg:tt)*) => {{
        use std::io::{self, Write};
        if $verbosity >= 999 {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            if let Err(e) = writeln!(handle, "{}", format!("c LOG {}", format_args!($($arg)*))) {
                die!("Failed to write message: {}", e);
            }
            if let Err(f) = handle.flush() {
                die!("Failed to flush stdout: {}", f);
            }
        }
    }}
}

#[cfg(not(feature = "logging"))]
macro_rules! LOG {
    ($($arg:tt)*) => {{}};
}

struct Config {
    input_path: String,
    output_path: String,
    proof_path: String,
    verbosity: i32,
    no_write: bool,
    size_limit: usize,
    occurrence_limit: usize,
}

fn average(a: usize, b: usize) -> f64 {
    if b != 0 {
        a as f64 / b as f64
    } else {
        0.0
    }
}

fn percent(a: usize, b: usize) -> f64 {
    100.0 * average(a, b)
}

struct Stats {
    added: usize,
    deleted: usize,
    eliminated: usize,
    parsed: usize,
    resolutions: usize,
    resolved: usize,
    rounds: usize,
    start_time: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Clause {
    id: usize,          // Clause identifier for debugging.
    literals: Vec<i32>, // Vector to store literals.
}

type ClauseRef = Rc<RefCell<Clause>>;

struct Matrix {
    matrix: Vec<Vec<ClauseRef>>,
}

impl Matrix {
    fn new() -> Self {
        Matrix { matrix: Vec::new() }
    }

    fn map_literal_to_index(&self, literal: i32) -> usize {
        // Optimization for matrix indexing
        // With this, lit and -lit will be next to each other
        if literal < 0 {
            (-literal * 2 - 2) as usize
        } else {
            (literal * 2 - 1) as usize
        }
    }

    fn init(&mut self, variables: usize, _verbosity: i32) {
        LOG!(
            _verbosity,
            "initializing matrix with {} variables",
            variables
        );
        self.matrix = vec![Vec::new(); 2 * variables];
    }
}

impl Index<i32> for Matrix {
    type Output = Vec<ClauseRef>;

    fn index(&self, literal: i32) -> &Self::Output {
        let computed_index = self.map_literal_to_index(literal);
        assert!(
            computed_index < self.matrix.len(),
            "Matrix index out of bounds"
        );
        &self.matrix[computed_index]
    }
}

impl IndexMut<i32> for Matrix {
    fn index_mut(&mut self, literal: i32) -> &mut Self::Output {
        let computed_index = self.map_literal_to_index(literal);
        assert!(
            computed_index < self.matrix.len(),
            "Matrix index out of bounds"
        );
        &mut self.matrix[computed_index]
    }
}

struct Marks {
    marks: Vec<bool>,
}

impl Marks {
    fn new() -> Self {
        Marks { marks: Vec::new() }
    }

    fn map_literal_to_index(&self, literal: i32) -> usize {
        // Optimization for indexing
        // With this, lit and -lit will be next to each other
        if literal < 0 {
            (-literal * 2 - 2) as usize
        } else {
            (literal * 2 - 1) as usize
        }
    }
    fn init(&mut self, variables: usize, _verbosity: i32) {
        LOG!(
            _verbosity,
            "initializing marks with {} variables",
            variables
        );
        self.marks = vec![false; 2 * variables];
    }

    fn mark(&mut self, literal: i32) {
        let computed_index = self.map_literal_to_index(literal);
        assert!(
            computed_index < self.marks.len(),
            "Marks index out of bounds"
        );
        self.marks[computed_index] = true;
    }

    fn unmark(&mut self, literal: i32) {
        let computed_index = self.map_literal_to_index(literal);
        assert!(
            computed_index < self.marks.len(),
            "Marks index out of bounds"
        );
        self.marks[computed_index] = false;
    }

    fn is_marked(&self, literal: i32) -> bool {
        let computed_index = self.map_literal_to_index(literal);
        assert!(
            computed_index < self.marks.len(),
            "Marks index out of bounds"
        );
        self.marks[computed_index]
    }
}

struct Values {
    values: Vec<i8>,
}

impl Index<i32> for Values {
    type Output = i8;

    fn index(&self, literal: i32) -> &Self::Output {
        let computed_index = self.map_literal_to_index(literal);
        assert!(
            computed_index < self.values.len(),
            "Values index out of bounds"
        );
        &self.values[computed_index]
    }
}

impl IndexMut<i32> for Values {
    fn index_mut(&mut self, literal: i32) -> &mut Self::Output {
        let computed_index = self.map_literal_to_index(literal);
        assert!(
            computed_index < self.values.len(),
            "Values index out of bounds"
        );
        &mut self.values[computed_index]
    }
}

impl Values {
    fn new() -> Self {
        Values { values: Vec::new() }
    }
    fn map_literal_to_index(&self, literal: i32) -> usize {
        // Optimization for indexing
        // With this, lit and -lit will be next to each other
        if literal < 0 {
            (-literal * 2 - 2) as usize
        } else {
            (literal * 2 - 1) as usize
        }
    }

    fn init(&mut self, variables: usize, _verbosity: i32) {
        LOG!(
            _verbosity,
            "initializing values with {} variables",
            variables
        );
        self.values = vec![0; 2 * variables];
    }
}

struct CNFFormula {
    variables: usize,
    clauses: LinkedList<ClauseRef>,
    found_empty_clause: bool,
    matrix: Matrix,
    marks: Marks,
    elimenated: Vec<bool>,
    values: Values,
    units: Vec<i32>,
    simplified: Vec<i32>,
}

impl CNFFormula {
    fn new() -> Self {
        CNFFormula {
            variables: 0,
            clauses: LinkedList::new(),
            found_empty_clause: false,
            matrix: Matrix::new(),
            marks: Marks::new(),
            elimenated: Vec::new(),
            values: Values::new(),
            units: Vec::new(),
            simplified: Vec::new(),
        }
    }

    fn connect_lit(&mut self, lit: i32, clause: ClauseRef, _verbosity: i32) {
        LOG!(
            _verbosity,
            "connecting literal {} to clause {:?}",
            lit,
            clause.borrow().id
        );
        self.matrix[lit].push(clause);
    }

    fn connect_clause(&mut self, clause: ClauseRef, _verbosity: i32) {
        LOG!(_verbosity, "connecting clause {:?}", clause.borrow().id);
        for &lit in &clause.borrow().literals {
            self.connect_lit(lit, clause.clone(), _verbosity);
        }
    }

    fn assign(&mut self, literal: i32) {
        LOG!(self.config.verbosity, "assigning literal {}", literal);
        self.values[literal] = 1;
        self.values[-literal] = -1;
        self.units.push(literal);
    }
}

struct SATContext {
    config: Config,
    formula: CNFFormula,
    stats: Stats,
    proof_file: Option<Box<dyn Write>>,
}

impl SATContext {
    fn new(config: Config) -> Self {
        SATContext {
            config,
            formula: CNFFormula::new(),
            stats: Stats {
                added: 0,
                deleted: 0,
                eliminated: 0,
                parsed: 0,
                resolutions: 0,
                resolved: 0,
                rounds: 0,
                start_time: Instant::now(),
            },
            proof_file: None,
        }
    }

    fn init(&mut self) {
        self.formula
            .marks
            .init(self.formula.variables, self.config.verbosity);
        self.formula
            .matrix
            .init(self.formula.variables, self.config.verbosity);
        self.formula.elimenated = vec![false; 1 + self.formula.variables];
        self.formula
            .values
            .init(self.formula.variables, self.config.verbosity);
    }
}

fn tautological_clause(ctx: &mut SATContext, clause: &Vec<i32>) -> bool {
    let mut res = false;
    for &lit in clause {
        if ctx.formula.marks.is_marked(lit) {
            continue;
        }
        if ctx.formula.values[lit] > 0 {
            LOG!(
                ctx.formula.config.verbosity,
                "tautological clause {:?} literal {} satisfied",
                clause,
                lit
            );
            res = true;
            break;
        }
        if ctx.formula.marks.is_marked(-lit) {
            LOG!(
                ctx.formula.config.verbosity,
                "tautological clause {:?} containing {} and {}",
                clause,
                lit,
                -lit
            );
            res = true;
            break;
        }
        ctx.formula.marks.mark(lit);
    }
    for &lit in clause {
        ctx.formula.marks.unmark(lit);
    }
    res
}

fn simplify_clause(ctx: &mut SATContext, clause: &Vec<i32>) -> Vec<i32> {
    let mut simplified = Vec::new();
    for &lit in clause {
        if ctx.formula.marks.is_marked(lit) {
            LOG!(
                ctx.formula.config.verbosity,
                "duplicated {} in {:?}",
                lit,
                clause
            );
            continue;
        }
        if ctx.formula.values[lit] < 0 {
            LOG!(
                ctx.formula.config.verbosity,
                "falsified {} in {:?}",
                lit,
                clause
            );
            continue;
        }
        assert!(!ctx.formula.marks.is_marked(-lit));
        ctx.formula.marks.mark(lit);
        simplified.push(lit);
    }
    for &lit in simplified.iter() {
        ctx.formula.marks.unmark(lit);
    }
    return simplified;
}

fn trace_added(ctx: &SATContext) {
    // TODO:
}

fn trace_deleted(ctx: &SATContext, clause: &Vec<i32>) {
    // TODO:
}

fn propagate(ctx: &mut SATContext) {
    if ctx.formula.found_empty_clause {
        return;
    }
    let mut propagated = 0;
    while propagated != ctx.formula.units.len() {
        let lit = ctx.formula.units[propagated];
        LOG!(formula.config.verbosity, "propagating literal {}", lit);
        propagated += 1;
        assert!(ctx.formula.values[lit] == 1);

        // Shrink clauses with `-lit`

        for clause_ref in ctx.formula.matrix[-lit].clone() {
            ctx.formula.simplified.clear();
            for &other in &clause_ref.borrow().literals {
                if other != -lit {
                    ctx.formula.simplified.push(other);
                }
            }
            LOG!(ctx.config.verbosity, "shrinking {:.?}", clause_ref.borrow());
            let new_size = clause_ref.borrow().literals.len() - 1;
            assert!(new_size == ctx.formula.simplified.len());
            if new_size == 0 {
                LOG!(
                    ctx.config.verbosity,
                    "conflicting {:.?}",
                    clause_ref.borrow()
                );
                ctx.formula.found_empty_clause = true;
            }
            trace_added(&ctx);
            if new_size == 0 {
                return;
            }
            trace_deleted(&ctx, &clause_ref.borrow().literals);
            clause_ref.borrow_mut().literals = ctx.formula.simplified.clone();
            LOG!(ctx.config.verbosity, "shrank to {:.?}", clause_ref.borrow());
            let unit = ctx.formula.simplified[0];
            let value = ctx.formula.values[unit];
            if value > 0 {
                continue;
            }
            if value < 0 {
                LOG!(
                    ctx.config.verbosity,
                    "conflicting clause after shrinking {:.?}",
                    clause_ref.borrow()
                );
                ctx.formula.found_empty_clause = true;
                return;
            }
            assert!(unit != 0);
        }
        ctx.formula.matrix[-lit].clear();

        // TODO Disconnect, dequeue, trace and delete satisfied clauses by 'lit'.
        for clause_ref in ctx.formula.matrix[lit].clone() {

            // TODO disconnect 'c' from 'matrix'.

            // TODO dequeue 'c' from 'clauses'.

            // TODO trace deletion and delete 'c'.
        }
        ctx.formula.matrix[lit].clear();
    }
}

fn parse_cnf(input_path: String, ctx: &mut SATContext) -> io::Result<()> {
    let path = Path::new(&input_path);
    let input: Box<dyn Read> = if input_path == "<stdin>" {
        message!(ctx.config.verbosity, "reading from '<stdin>'");
        Box::new(io::stdin())
    } else {
        message!(ctx.config.verbosity, "reading from '{}'", input_path);
        let file = File::open(&input_path)?;
        if path.extension().unwrap() == "bz2" {
            LOG!(ctx.config.verbosity, "reading BZ2 compressed file");
            Box::new(BzDecoder::new(file))
        } else if path.extension().unwrap() == "gz" {
            LOG!(ctx.config.verbosity, "reading GZ compressed file");
            Box::new(GzDecoder::new(file))
        } else if path.extension().unwrap() == "xz" {
            LOG!(ctx.config.verbosity, "reading XZ compressed file");
            Box::new(XzDecoder::new(file))
        } else {
            LOG!(ctx.config.verbosity, "reading uncompressed file");
            Box::new(file)
        }
    };

    let reader = BufReader::new(input);
    let mut header_parsed = false;
    let mut line_number = 0;

    for line in reader.lines() {
        line_number += 1;
        let line = line?;
        if line.starts_with('c') {
            continue; // Skip comment lines
        }
        if line.starts_with("p cnf") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                parse_error!(ctx, "Invalid header format.", line_number);
            }
            ctx.formula.variables = parts[2].parse().unwrap_or_else(|_| {
                parse_error!(ctx, "Could not read number of variables.", line_number);
            });
            ctx.init(); // TODO: Merge with above
            let clauses_count: usize = match parts[3].parse() {
                Ok(num) => num,
                Err(_) => parse_error!(ctx, "Could not read number of clauses.", line_number),
            };
            header_parsed = true;
            message!(
                ctx.config.verbosity,
                "parsed 'p cnf {} {}' header",
                ctx.formula.variables,
                clauses_count
            );
        } else if header_parsed {
            let clause: Vec<i32> = line
                .split_whitespace()
                .map(|num| {
                    num.parse().unwrap_or_else(|_| {
                        parse_error!(ctx, "Invalid literal format.", line_number);
                    })
                })
                .filter(|&x| x != 0)
                .collect();
            LOG!(ctx.config.verbosity, "parsed clause: {:?}", clause);
            ctx.stats.parsed += 1;

            if !ctx.formula.found_empty_clause && !tautological_clause(ctx, &clause) {
                let simplified_clause = simplify_clause(ctx, &clause);
                if clause.len() != simplified_clause.len() {
                    LOG!(
                        ctx.config.verbosity,
                        "simplified clause: {:?}",
                        simplified_clause
                    );
                    trace_added(ctx);
                    trace_deleted(ctx, &clause);
                }
                match simplified_clause.len() {
                    0 => {
                        if !ctx.formula.found_empty_clause {
                            verbose!(ctx.config.verbosity, 2, "found empty clause");
                            ctx.formula.found_empty_clause = true;
                        }
                    }
                    1 => {
                        LOG!(ctx.config.verbosity, "unit clause: {:?}", simplified_clause);
                        ctx.formula.assign(simplified_clause[0]);
                        propagate(ctx);
                    }
                    _ => {
                        let new_clause = Rc::new(RefCell::new(Clause {
                            id: ctx.stats.added,
                            literals: ctx.formula.simplified.clone(),
                        }));
                        ctx.formula
                            .connect_clause(new_clause.clone(), ctx.config.verbosity);
                        ctx.formula.clauses.push_back(new_clause);
                        ctx.stats.added += 1;
                    }
                }
            }
        } else {
            parse_error!(ctx, "CNF header not found.", line_number);
        }
    }
    verbose!(
        ctx.config.verbosity,
        1,
        "parsed {} clauses",
        ctx.stats.parsed
    );
    Ok(())
}

fn size_or_occurrence_limit_exceeded(ctx: &SATContext, pivot: i32) -> bool {
    for clause_ref in ctx.formula.matrix[pivot].clone() {
        if clause_ref.borrow().literals.len() > ctx.config.size_limit {
            return true;
        }
        for &lit in &clause_ref.borrow().literals {
            if occurrences(ctx, lit) > ctx.config.occurrence_limit {
                return true;
            }
        }
    }
    false
}

fn can_eliminate_variable(ctx: &SATContext, pivot: i32) -> bool {
    assert!(pivot > 0);
    if ctx.formula.elimenated[pivot as usize] {
        return false;
    }
    if ctx.formula.values[pivot] != 0 {
        return false;
    }
    let pos = occurrences(ctx, pivot);
    if pos > ctx.config.occurrence_limit {
        return false;
    }
    let neg = occurrences(ctx, -pivot);
    if neg > ctx.config.occurrence_limit {
        return false;
    }
    if size_or_occurrence_limit_exceeded(ctx, pivot) {
        return false;
    }
    if size_or_occurrence_limit_exceeded(ctx, -pivot) {
        return false;
    }
    let limit = pos + neg;
    LOG!(
        ctx.config.verbosity,
        "trying to eliminate variable {} with {} occurrences",
        pivot,
        limit
    );

    // TODO got over all clauses with 'pivot' and count how often
    // resolving them with a clause with '-pivot' produces an
    // non-tautological resolvent using 'can_be_resolved'.

    false
}

fn add_resolvent(ctx: &mut SATContext, clause_ref: ClauseRef, other_ref: ClauseRef) {
    // TODO if 'c' can be resolved with 'd' on 'pivot' (assume
    // 'pivot' is in 'c' and '-pivot' in 'd'), i.e., the resolvent is not
    // tautological, then simplify and add it as new clause.
    // Also connect and enqueue and trace it.

    LOG!(
        "clause {:?} resolving 1st {} antecedent",
        clause_ref.borrow(),
        pivot
    );
    LOG!(
        "clause {:?} resolving 2nd {} antecedent",
        other_ref.borrow(),
        -pivot
    );
}

fn add_all_resolvents(ctx: &mut SATContext, pivot: i32) {
    for clause_ref in ctx.formula.matrix[pivot].clone() {
        for other_ref in ctx.formula.matrix[-pivot].clone() {
            add_resolvent(ctx, clause_ref.clone(), other_ref.clone());
        }
    }
}

fn disconnect_and_delete_all_clause_with_literal(ctx: &mut SATContext, lit: i32) {

    // TODO disconnect, trace deletion and delete all clauses with 'lit'.

    // NOTE you cannot disconnect the literal from 'matrix[lit]' while
    // traversing it, so be careful (which makes the code ugly). See also
    // 'disconnect_clause (..., int except)' above.

    // TODO also dequeue clause from 'clauses'.

    // TODO and finally delete it.
}

fn remove_all_clauses_with_variable(ctx: &mut SATContext, pivot: i32) {
    disconnect_and_delete_all_clause_with_literal(ctx, pivot);
    disconnect_and_delete_all_clause_with_literal(ctx, -pivot);
}

// TODO First get the basic 'eliminate' working.

// TODO+ Second you might want to sort variables to be eliminated by
// their occurrences (using a separate 'candidates' stack) and eliminate
// variables with few occurrences first.

// TODO++ Third you could implement a working queue instead of a round
// based scheme, that is you mark variables as candidates if they occur
// in deleted clauses.  Initially all variables are marked.  This would
// avoid going through too many useless elimination attempts.

// TODO+++ Finally try adding backward subsumption if you feel fancy.

// But at least try to get basic eliminate working as is, which should
// only require to work on the other 'TODO' (without '+'s at the end).

fn eliminate(ctx: &mut SATContext) {
    let start_time = Instant::now();
    while !ctx.formula.found_empty_clause {
        ctx.stats.rounds += 1;
        verbose!(
            ctx.config.verbosity,
            1,
            "starting variable elimination round {}",
            ctx.stats.rounds
        );
        let before = ctx.stats.eliminated;
        for pivot in 1..=ctx.formula.variables + 1 {
            if ctx.formula.found_empty_clause {
                break;
            }
            if can_eliminate_variable(ctx, pivot as i32) {
                add_all_resolvents(ctx, pivot as i32);
                remove_all_clauses_with_variable(ctx, pivot as i32);
                ctx.formula.elimenated[pivot] = true;
                ctx.stats.eliminated += 1;
                propagate(ctx);
            }
            let after = ctx.stats.eliminated;
            if before == after {
                verbose!(
                    ctx.config.verbosity,
                    1,
                    "unsuccesful variable elimination round {}",
                    ctx.stats.rounds
                );
                break;
            }
            let delta = after - before;
            verbose!(
                ctx.config.verbosity,
                1,
                "eliminated {} variables {} in round {}",
                delta,
                percent(delta, ctx.formula.variables),
                ctx.stats.rounds
            );
        }
    }
    message!(
        ctx.config.verbosity,
        "eliminated {} variables {} in {:?} and {} rounds",
        ctx.stats.eliminated,
        percent(ctx.stats.eliminated, ctx.formula.variables),
        Instant::now() - start_time,
        ctx.stats.rounds
    );
}

fn print(ctx: &SATContext) {
    if ctx.config.no_write {
        return;
    }

    let start_time = Instant::now();

    let output_path = &ctx.config.output_path;
    let mut output: Box<dyn Write> = if output_path == "<stdout>" {
        Box::new(io::stdout())
    } else {
        match output_path.as_str() {
            path if path.ends_with(".bz2") => {
                let file = File::create(path).expect("Failed to create output file");
                Box::new(BzEncoder::new(file, bzip2::Compression::default()))
            }
            path if path.ends_with(".gz") => {
                let file = File::create(path).expect("Failed to create output file");
                Box::new(GzEncoder::new(file, flate2::Compression::default()))
            }
            path if path.ends_with(".xz") => {
                let file = File::create(path).expect("Failed to create output file");
                Box::new(XzEncoder::new(file, 6)) // Compression level set to 6
            }
            path => Box::new(File::create(path).expect("Failed to create output file")),
        }
    };

    message!(
        ctx.config.verbosity,
        "writing simplified formula to '{}'",
        ctx.config.output_path
    );

    if ctx.formula.found_empty_clause {
        writeln!(output, "p cnf {} 0", ctx.formula.variables).expect("Failed to write CNF header");
    } else {
        writeln!(
            output,
            "p cnf {} {}",
            ctx.formula.variables,
            ctx.formula.clauses.len()
        )
        .expect("Failed to write CNF header");
        for clause in &ctx.formula.clauses {
            let literals = clause
                .borrow()
                .literals
                .iter()
                .map(|lit| lit.to_string())
                .collect::<Vec<String>>()
                .join(" ");
            writeln!(output, "{} 0", literals).expect("Failed to write clause");
        }
    }

    if let Err(e) = output.flush() {
        die!("Failed to flush output: {}", e);
    }

    message!(
        ctx.config.verbosity,
        "writing took {:?}",
        Instant::now() - start_time
    );
}

fn report(ctx: &SATContext) {
    let elapsed_time = ctx.stats.start_time.elapsed().as_secs_f64();
    let propagated_units = ctx.formula.units.len();
    let simplified_clauses = ctx.stats.parsed - ctx.stats.added + ctx.stats.deleted;

    message!(
        ctx.config.verbosity,
        "{:<20} {:>10}    {:.2}% variables",
        "propagated-units:",
        propagated_units,
        percent(propagated_units, ctx.formula.variables)
    );
    message!(
        ctx.config.verbosity,
        "{:<20} {:>10}",
        "elimination-rounds:",
        ctx.stats.rounds
    );
    message!(
        ctx.config.verbosity,
        "{:<20} {:>10}    {:.2}% variables",
        "eliminated-variables:",
        ctx.stats.eliminated,
        percent(ctx.stats.eliminated, ctx.formula.variables)
    );
    message!(
        ctx.config.verbosity,
        "{:<20} {:>10}    {:.2}% clauses",
        "simplified-clauses:",
        simplified_clauses,
        percent(simplified_clauses, ctx.stats.parsed)
    );
    message!(
        ctx.config.verbosity,
        "{:<20} {:13.2} seconds",
        "process-time:",
        elapsed_time
    );
}

fn prove(ctx: &mut SATContext) {
    if ctx.config.proof_path.is_empty() {
        return;
    }

    let proof_path = &ctx.config.proof_path;
    let proof_file: Box<dyn Write> = if proof_path == "<stdout>" {
        Box::new(io::stdout())
    } else {
        match proof_path.as_str() {
            path if path.ends_with(".bz2") => {
                let file = File::create(path).expect("Failed to create proof file");
                Box::new(BzEncoder::new(file, bzip2::Compression::default()))
            }
            path if path.ends_with(".gz") => {
                let file = File::create(path).expect("Failed to create proof file");
                Box::new(GzEncoder::new(file, flate2::Compression::default()))
            }
            path if path.ends_with(".xz") => {
                let file = File::create(path).expect("Failed to create proof file");
                Box::new(XzEncoder::new(file, 6)) // Compression level set to 6
            }
            path => Box::new(File::create(path).expect("Failed to create proof file")),
        }
    };

    ctx.proof_file = Some(proof_file);
    message!(ctx.config.verbosity, "writing proof to '{}'", proof_path);
}

fn occurrences(ctx: &SATContext, lit: i32) -> usize {
    ctx.formula.matrix[lit].len()
}

fn parse_arguments() -> Config {
    let app = Command::new("BabySub")
        .version("1.0")
        .author("Bernhard Gstrein")
        .about("Processes and simplifies logical formulae in DIMACS CNF format.")
        .arg(
            Arg::new("input")
                .help("Sets the input file to use")
                .index(1),
        )
        .arg(
            Arg::new("output")
                .help("Sets the output file to use")
                .index(2),
        )
        .arg(
            Arg::new("proof")
                .help("Sets the proof file to use")
                .index(3),
        )
        .arg(
            Arg::new("verbosity")
                .short('v')
                .action(ArgAction::Count)
                .help("Increases verbosity level"),
        )
        .arg(
            Arg::new("size-limit")
                .short('s')
                .value_parser(value_parser!(usize))
                .default_value("1000")
                .help("Set the size limit"),
        )
        .arg(
            Arg::new("occurrence-limit")
                .short('o')
                .value_parser(value_parser!(usize))
                .default_value("10000")
                .help("Set the occurrence limit"),
        )
        .arg(Arg::new("quiet").short('q').help("Suppresses all output"))
        .arg(Arg::new("no-output").short('n').help("Do not write output"));
    #[cfg(feature = "logging")]
    let app = app.arg(
        Arg::new("logging")
            .short('l')
            .help("Enables detailed logging for debugging")
            .action(ArgAction::SetTrue),
    );

    let matches = app.get_matches();

    #[cfg(not(feature = "logging"))]
    let verbosity = if matches.is_present("quiet") {
        -1
    } else {
        *matches.get_one::<u8>("verbosity").unwrap_or(&0) as i32
    };

    #[cfg(feature = "logging")]
    let verbosity = if matches.is_present("quiet") {
        -1
    } else if matches.get_flag("logging") {
        999
    } else {
        *matches.get_one::<u8>("verbosity").unwrap_or(&0) as i32
    };

    Config {
        input_path: matches.value_of("input").unwrap_or("<stdin>").to_string(),
        output_path: matches.value_of("output").unwrap_or("<stdout>").to_string(),
        proof_path: matches.value_of("proof").unwrap_or("").to_string(),
        verbosity,
        no_write: matches.is_present("no-output"),
        size_limit: *matches.get_one("size-limit").unwrap(),
        occurrence_limit: *matches.get_one("occurrence-limit").unwrap(),
    }
}

fn setup_context(config: Config) -> SATContext {
    let ctx = SATContext::new(config);
    message!(ctx.config.verbosity, "BabySub Subsumption Preprocessor");
    ctx
}

fn main() {
    let config = parse_arguments();
    let mut ctx = setup_context(config);
    prove(&mut ctx);

    if let Err(e) = parse_cnf(ctx.config.input_path.clone(), &mut ctx) {
        die!("Failed to parse CNF: {}", e);
    }

    eliminate(&mut ctx);
    print(&ctx);
    report(&ctx);
}
