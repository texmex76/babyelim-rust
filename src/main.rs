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
    bound_limit: usize,
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
    eliminated: Vec<bool>,
    candidates: Vec<i32>,
    rescheduled: Vec<bool>,
    values: Values,
    units: Vec<i32>,
    simplified: Vec<i32>,
    propagated: usize,
}

impl CNFFormula {
    fn new() -> Self {
        CNFFormula {
            variables: 0,
            clauses: LinkedList::new(),
            found_empty_clause: false,
            matrix: Matrix::new(),
            marks: Marks::new(),
            eliminated: Vec::new(),
            candidates: Vec::new(),
            rescheduled: Vec::new(),
            values: Values::new(),
            units: Vec::new(),
            simplified: Vec::new(),
            propagated: 0,
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

    fn disconnect_clause_except(&mut self, clause: ClauseRef, lit: i32, _verbosity: i32) {
        LOG!(
            _verbosity,
            "disconnecting clause {:?} except literal {}",
            clause.borrow().id,
            lit
        );
        for &other in &clause.borrow().literals {
            if other != lit {
                self.matrix[other].retain(|c| c != &clause);
            }
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
        self.formula.eliminated = vec![false; 1 + self.formula.variables];
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

fn trace_clause(ctx: &mut SATContext, prefix: String, clause: &Vec<i32>) {
    if ctx.proof_file.is_none() {
        return;
    }
    if prefix != "" {
        write!(ctx.proof_file.as_mut().unwrap(), "{}", prefix).unwrap();
    }
    for &lit in clause {
        write!(ctx.proof_file.as_mut().unwrap(), "{} ", lit).unwrap();
    }
    write!(ctx.proof_file.as_mut().unwrap(), "0\n").unwrap();
}

fn trace_deleted(ctx: &mut SATContext, clause: &Vec<i32>) {
    trace_clause(ctx, "d ".to_string(), clause);
}

fn trace_added(ctx: &mut SATContext, clause: &Vec<i32>) {
    trace_clause(ctx, "".to_string(), clause);
}

fn propagate(ctx: &mut SATContext, flush_units: bool) {
    if ctx.formula.found_empty_clause {
        return;
    }
    while ctx.formula.propagated != ctx.formula.units.len() {
        let lit = ctx.formula.units[ctx.formula.propagated];
        LOG!(formula.config.verbosity, "propagating literal {}", lit);
        ctx.formula.propagated += 1;
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
            trace_added(ctx, &ctx.formula.simplified.clone());
            trace_deleted(ctx, &clause_ref.borrow().literals);
            clause_ref.borrow_mut().literals = ctx.formula.simplified.clone();
            LOG!(ctx.config.verbosity, "shrank to {:.?}", clause_ref.borrow());
            if new_size == 0 {
                if !ctx.formula.found_empty_clause {
                    verbose!(ctx.config.verbosity, 2, "found empty clause");
                    ctx.formula.found_empty_clause = true;
                }
            } else if new_size == 1 {
                let unit = ctx.formula.simplified[0];
                let value = ctx.formula.values[unit];
                if value == 0 {
                    ctx.formula.assign(unit);
                }
            }
        }
        ctx.formula.matrix[-lit].clear();

        // Disconnect, dequeue, trace and delete satisfied clauses by 'lit'.
        let mut skipped = Rc::new(RefCell::new(Clause {
            id: usize::MAX,
            literals: Vec::new(),
        }));
        for clause_ref in ctx.formula.matrix[lit].clone() {
            if clause_ref.borrow().literals.len() > 1 || skipped.borrow().id != usize::MAX {
                disconnect_dequeue_trace_and_delete_clause(ctx, clause_ref, lit);
            } else {
                skipped = clause_ref;
            }
        }
        assert!(skipped.borrow().id != usize::MAX);
        ctx.formula.matrix[lit].clear();
        if !flush_units {
            disconnect_dequeue_trace_and_delete_clause(ctx, skipped, lit);
        } else {
            ctx.formula.matrix[lit].push(skipped);
        }
    }
}

fn flush_unit_clauses(ctx: &mut SATContext) {
    verbose!(
        ctx.config.verbosity,
        1,
        "flushing {} unit clauses",
        ctx.formula.units.len()
    );
    // doint this to avoid simultaneous mutable and immutable borrows
    // of ctx
    // works but feels clunky
    let mut clauses_and_units = vec![];
    for &unit in &ctx.formula.units {
        assert!(ctx.formula.matrix[-unit].is_empty());
        assert!(ctx.formula.matrix[unit].len() == 1);
        for clause_ref in ctx.formula.matrix[unit].clone() {
            clauses_and_units.push((clause_ref, unit));
        }
    }
    for (clause_ref, unit) in clauses_and_units {
        disconnect_dequeue_trace_and_delete_clause(ctx, clause_ref, unit);
        ctx.formula.matrix[unit].clear();
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
                    trace_added(ctx, &simplified_clause);
                    trace_deleted(ctx, &clause);
                }
                ctx.stats.added += 1;
                let new_clause = Rc::new(RefCell::new(Clause {
                    id: ctx.stats.added,
                    literals: ctx.formula.simplified.clone(),
                }));
                ctx.formula
                    .connect_clause(new_clause.clone(), ctx.config.verbosity);
                ctx.formula.clauses.push_back(new_clause.clone());
                if new_clause.borrow().literals.is_empty() {
                    if !ctx.formula.found_empty_clause {
                        verbose!(ctx.config.verbosity, 2, "found empty clause");
                        ctx.formula.found_empty_clause = true;
                    }
                } else if new_clause.borrow().literals.len() == 1 {
                    let unit = new_clause.borrow().literals[0];
                    LOG!(ctx.config.verbosity, "unit clause: {:?}", unit);
                    ctx.formula.assign(unit);
                    propagate(ctx, false); // Propagate but delay flushing
                }
            } else {
                trace_deleted(ctx, &clause);
            }
        } else {
            parse_error!(ctx, "CNF header not found.", line_number);
        }
    }
    flush_unit_clauses(ctx);
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

fn can_be_resolved(
    ctx: &mut SATContext,
    pivot: i32,
    clause_ref: ClauseRef,
    other_ref: ClauseRef,
) -> bool {
    LOG!(
        "clause {:?} tryint to resolve 1st {} antecedent",
        clause_ref.borrow(),
        pivot
    );
    LOG!(
        "clause {:?} trying to resolve 2nd {} antecedent",
        other_ref.borrow(),
        -pivot
    );
    let mut resolvent = Vec::new();
    for &lit in &clause_ref.borrow().literals {
        if lit != pivot {
            resolvent.push(lit);
        }
    }
    for &lit in &other_ref.borrow().literals {
        if lit != -pivot {
            resolvent.push(lit);
        }
    }
    if tautological_clause(ctx, &resolvent) {
        return false;
    }
    LOG!("resolvent not tautological");
    true
}

fn can_eliminate_variable(ctx: &mut SATContext, pivot: i32) -> bool {
    assert!(pivot > 0);
    if ctx.formula.eliminated[pivot as usize] {
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
    let limit = pos + neg + ctx.config.bound_limit;
    LOG!(
        ctx.config.verbosity,
        "trying to eliminate variable {} with {} occurrences",
        pivot,
        limit
    );

    let mut resolvents = 0;
    for clause_ref in ctx.formula.matrix[pivot].clone() {
        for other_ref in ctx.formula.matrix[-pivot].clone() {
            resolvents += 1;
            if can_be_resolved(ctx, pivot, clause_ref.clone(), other_ref.clone())
                && resolvents > limit
            {
                LOG!("variable {} produces more than {} resolvents", pivot, limit);
                return false;
            }
        }
    }

    LOG!("variable {} produces {} resolvents", pivot, resolvents);
    true
}

fn add_resolvent(ctx: &mut SATContext, pivot: i32, clause_ref: ClauseRef, other_ref: ClauseRef) {
    let mut resolvent = Vec::new();
    for &lit in &clause_ref.borrow().literals {
        if lit != pivot {
            resolvent.push(lit);
        }
    }
    for &lit in &other_ref.borrow().literals {
        if lit != -pivot {
            resolvent.push(lit);
        }
    }
    if tautological_clause(ctx, &resolvent) {
        return;
    }
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
    simplify_clause(ctx, &resolvent);
    ctx.stats.added += 1;
    let new_clause = Rc::new(RefCell::new(Clause {
        id: ctx.stats.added,
        literals: resolvent,
    }));
    ctx.formula
        .connect_clause(new_clause.clone(), ctx.config.verbosity);
    ctx.formula.clauses.push_back(new_clause.clone());
    trace_added(ctx, &new_clause.borrow().literals);
}

fn add_all_resolvents(ctx: &mut SATContext, pivot: i32) {
    for clause_ref in ctx.formula.matrix[pivot].clone() {
        for other_ref in ctx.formula.matrix[-pivot].clone() {
            add_resolvent(ctx, pivot, clause_ref.clone(), other_ref.clone());
        }
    }
}

fn disconnect_and_delete_all_clause_with_literal(ctx: &mut SATContext, lit: i32) {
    for clause_ref in ctx.formula.matrix[lit].clone() {
        ctx.formula
            .disconnect_clause_except(clause_ref.clone(), lit, ctx.config.verbosity);
        // INFO: dequeue from clauses. really messy without unsafe code
        let index = ctx
            .formula
            .clauses
            .iter()
            .position(|x| Rc::ptr_eq(x, &clause_ref))
            .unwrap();
        let mut split_list = ctx.formula.clauses.split_off(index);
        split_list.pop_front();
        ctx.formula.clauses.append(&mut split_list);
        // end dequeue
        trace_deleted(ctx, &clause_ref.borrow().literals);
    }
    ctx.formula.matrix[lit].clear();
}

fn disconnect_dequeue_trace_and_delete_clause(
    ctx: &mut SATContext,
    clause_ref: ClauseRef,
    except: i32,
) {
    ctx.formula
        .disconnect_clause_except(clause_ref.clone(), except, ctx.config.verbosity);
    // INFO: dequeue from clauses. really messy without unsafe code
    let index = ctx
        .formula
        .clauses
        .iter()
        .position(|x| Rc::ptr_eq(x, &clause_ref))
        .unwrap();
    let mut split_list = ctx.formula.clauses.split_off(index);
    split_list.pop_front();
    ctx.formula.clauses.append(&mut split_list);
    // end dequeue
    trace_deleted(ctx, &clause_ref.borrow().literals);
}

fn remove_all_clauses_with_variable(ctx: &mut SATContext, pivot: i32) {
    disconnect_and_delete_all_clause_with_literal(ctx, pivot);
    disconnect_and_delete_all_clause_with_literal(ctx, -pivot);
}

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
        assert!(ctx.formula.candidates.is_empty());
        for pivot in 1..=ctx.formula.variables + 1 {
            if !ctx.formula.eliminated[pivot]
                && ctx.formula.values[pivot as i32] == 0
                && ctx.formula.rescheduled[pivot]
            {
                ctx.formula.candidates.push(pivot as i32);
            }
        }
        ctx.formula.candidates.sort_by(|a, b| {
            ctx.formula.matrix[*a]
                .len()
                .cmp(&ctx.formula.matrix[*b].len())
        });
        verbose!(
            ctx.config.verbosity,
            1,
            "scheduled {} variables {}% in round {}",
            ctx.formula.candidates.len(),
            percent(ctx.formula.candidates.len(), ctx.formula.variables),
            ctx.stats.rounds
        );
        let before = ctx.stats.eliminated;
        while !ctx.formula.found_empty_clause && !ctx.formula.candidates.is_empty() {
            let pivot = ctx.formula.candidates.pop().unwrap();
            ctx.formula.rescheduled[pivot as usize] = false;
            if !can_eliminate_variable(ctx, pivot) {
                continue;
            }
            add_all_resolvents(ctx, pivot);
            ctx.formula.eliminated[pivot as usize] = true;
            remove_all_clauses_with_variable(ctx, pivot as i32);
            ctx.stats.eliminated += 1;
            propagate(ctx, true);
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
    message!(
        ctx.config.verbosity,
        "eliminated {} variables {} in {:?} and {} rounds",
        ctx.stats.eliminated,
        percent(ctx.stats.eliminated, ctx.formula.variables),
        Instant::now() - start_time,
        ctx.stats.rounds
    );
}

fn print(ctx: &mut SATContext) {
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
        if !ctx.config.proof_path.is_empty() {
            let mut skipped_first_empty_clause = false;
            // Doing this to avois simultaneous mutable and immutable borrows
            let mut literals_to_trace = vec![];

            for clause in &ctx.formula.clauses {
                if clause.borrow().literals.len() > 0 {
                    literals_to_trace.push(clause.borrow().literals.clone());
                } else if skipped_first_empty_clause {
                    literals_to_trace.push(clause.borrow().literals.clone());
                } else {
                    skipped_first_empty_clause = true;
                }
            }
            assert!(skipped_first_empty_clause);

            for literals in literals_to_trace {
                trace_deleted(ctx, &literals);
            }
        }
        writeln!(output, "p cnf {} 0", ctx.formula.variables,).expect("Failed to write CNF header");
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
    assert!(ctx.stats.added >= ctx.stats.deleted);
    let elapsed_time = ctx.stats.start_time.elapsed().as_secs_f64();
    let simplified_clauses = ctx.stats.parsed - ctx.stats.added + ctx.stats.deleted;

    message!(
        ctx.config.verbosity,
        "{:<20} {:>10}    {:.2}% variables",
        "propagated-units:",
        ctx.formula.propagated,
        percent(ctx.formula.propagated, ctx.formula.variables)
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
    let app = Command::new("BabyElim")
        .version("1.0")
        .author("Bernhard Gstrein")
        .about("Processes and simplifies logical formulae in DIMACS CNF format.")
        .arg(
            Arg::new("input")
                .help("Sets the input file to use")
                .index(1),
        )
        .arg(
            Arg::new("proof")
                .help("Sets the proof file to use")
                .index(2),
        )
        .arg(
            Arg::new("output")
                .help("Sets the output file to use")
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
        .arg(
            Arg::new("bound-limit")
                .short('b')
                .value_parser(value_parser!(usize))
                .default_value("0")
                .help("Bount on number of added clauses per elimination"),
        )
        .arg(Arg::new("quiet").short('q').help("Suppresses all output"))
        .arg(
            Arg::new("force-proof-writing")
                .short('f')
                .help("Force proof writing"),
        )
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

    let force_proof_writing = matches.is_present("force-proof-writing");
    let proof_path = matches.value_of("proof").unwrap_or("").to_string();
    let path = Path::new(&proof_path);
    if path.exists()
        && !force_proof_writing
        && (path.extension().unwrap() == "cnf"
            || path.extension().unwrap() == "cnf.bz2"
            || path.extension().unwrap() == "cnf.gz"
            || path.extension().unwrap() == "cnf.xz")
    {
        die!("Proof file already exists: '{}'", proof_path);
    }

    Config {
        input_path: matches.value_of("input").unwrap_or("<stdin>").to_string(),
        output_path: matches.value_of("output").unwrap_or("<stdout>").to_string(),
        proof_path,
        verbosity,
        no_write: matches.is_present("no-output"),
        size_limit: *matches.get_one("size-limit").unwrap(),
        occurrence_limit: *matches.get_one("occurrence-limit").unwrap(),
        bound_limit: *matches.get_one("bound-limit").unwrap(),
    }
}

fn setup_context(config: Config) -> SATContext {
    let ctx = SATContext::new(config);
    message!(
        ctx.config.verbosity,
        "BabyElim Variable Elimination Preprocessor"
    );
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
    print(&mut ctx);
    report(&ctx);
}
