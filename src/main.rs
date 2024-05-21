use bzip2;
use bzip2::read::BzDecoder;
use bzip2::write::BzEncoder;
use clap::{Arg, ArgAction, Command};
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
    verbosity: i32,
    backward_mode: bool,
    sign: bool,
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
    added_clauses: usize,
    clauses: LinkedList<ClauseRef>,
    found_empty_clause: bool,
    matrix: Matrix,
    marks: Marks,
    elimenated: Vec<bool>,
    values: Values,
    units: Vec<i32>,
    resolvent: Vec<i32>,
    simplified: Vec<i32>,
}

impl CNFFormula {
    fn new() -> Self {
        CNFFormula {
            variables: 0,
            added_clauses: 0,
            clauses: LinkedList::new(),
            found_empty_clause: false,
            matrix: Matrix::new(),
            marks: Marks::new(),
            elimenated: Vec::new(),
            values: Values::new(),
            units: Vec::new(),
            resolvent: Vec::new(),
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

fn report_stats(ctx: &mut SATContext) {
    // let elapsed_time = ctx.stats.start_time.elapsed().as_secs_f64();
    // message!(
    //     ctx.config.verbosity,
    //     "{:<20} {:>10}    clauses {:.2} per subsumed",
    //     "checked:",
    //     ctx.stats.checked,
    //     average(ctx.stats.subsumed, ctx.stats.subsumed)
    // );
    // message!(
    //     ctx.config.verbosity,
    //     "{:<20} {:>10}    clauses {:.0}%",
    //     "subsumed:",
    //     ctx.stats.subsumed,
    //     percent(ctx.stats.subsumed, ctx.stats.parsed)
    // );
    // message!(
    //     ctx.config.verbosity,
    //     "{:<20} {:13.2} seconds",
    //     "process-time:",
    //     elapsed_time
    // );
    // TODO: Port from C++ implementation
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

fn eliminate(ctx: &mut SATContext) {
    // TODO:
}

fn print(ctx: &mut SATContext) {
    // TODO:
    // let mut output: Box<dyn Write> = if ctx.config.output_path == "<stdout>" {
    //     Box::new(io::stdout())
    // } else {
    //     match ctx.config.output_path.as_str() {
    //         path if path.ends_with(".bz2") => {
    //             let file = File::create(path).expect("Failed to create output file");
    //             Box::new(BzEncoder::new(file, bzip2::Compression::default()))
    //         }
    //         path if path.ends_with(".gz") => {
    //             let file = File::create(path).expect("Failed to create output file");
    //             Box::new(GzEncoder::new(file, flate2::Compression::default()))
    //         }
    //         path if path.ends_with(".xz") => {
    //             let file = File::create(path).expect("Failed to create output file");
    //             Box::new(XzEncoder::new(file, 6)) // Compression level set to 6
    //         }
    //         path => Box::new(File::create(path).expect("Failed to create output file")),
    //     }
    // };
    //
    // writeln!(
    //     output,
    //     "p cnf {} {}",
    //     ctx.formula.variables,
    //     ctx.formula.clauses.len()
    // )
    // .expect("Failed to write CNF header");
    //
    // for clause in &ctx.formula.clauses {
    //     let literals = clause
    //         .literals
    //         .iter()
    //         .map(|lit| lit.to_string())
    //         .collect::<Vec<String>>()
    //         .join(" ");
    //     writeln!(output, "{} 0", literals).expect("Failed to write clause");
    // }
    //
    // match output.flush() {
    //     Ok(_) => (),
    //     Err(e) => die!("Failed to flush output: {}", e),
    // }
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
            Arg::new("verbosity")
                .short('v')
                .action(ArgAction::Count)
                .help("Increases verbosity level"),
        )
        .arg(Arg::new("quiet").short('q').help("Suppresses all output"))
        .arg(
            Arg::new("forward-mode")
                .short('f')
                .help("Enables forward subsumption"),
        )
        .arg(
            Arg::new("backward-mode")
                .short('b')
                .help("Enables backward subsumption"),
        )
        .arg(
            Arg::new("sign")
                .short('s')
                .help("Computes and adds a hash signature to the output"),
        );

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

    if matches.is_present("forward-mode") && matches.is_present("backward-mode") {
        die!("Cannot enable both forward and backward subsumption");
    }

    Config {
        input_path: matches.value_of("input").unwrap_or("<stdin>").to_string(),
        output_path: matches.value_of("output").unwrap_or("<stdout>").to_string(),
        verbosity,
        backward_mode: matches.is_present("backward-mode"),
        sign: matches.is_present("sign"),
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

    if let Err(e) = parse_cnf(ctx.config.input_path.clone(), &mut ctx) {
        die!("Failed to parse CNF: {}", e);
    }

    eliminate(&mut ctx);
    print(&mut ctx);
    report_stats(&mut ctx);
}
