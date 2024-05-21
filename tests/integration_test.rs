use assert_cmd::Command;
use std::fs;
use std::io::Write;
use std::process::Output;

// Path constants for the tests directory and test files
const TEST_DIR: &str = "tests/test_cases";
const CNF_EXT: &str = "cnf";
const EXECUTABLE_NAME: &str = "babyelim-rust";

fn run_test_case(test_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let current_dir = std::env::current_dir().unwrap();
    let cnf_path = current_dir
        .join(TEST_DIR)
        .join(test_name)
        .with_extension(CNF_EXT);
    let log_path = current_dir
        .join(TEST_DIR)
        .join(test_name)
        .with_extension("log");
    let err_path = current_dir
        .join(TEST_DIR)
        .join(test_name)
        .with_extension("err");
    let output_path = current_dir
        .join(TEST_DIR)
        .join(test_name)
        .with_extension("out");
    let prf_path = current_dir
        .join(TEST_DIR)
        .join(test_name)
        .with_extension("prf");
    let sol_path = current_dir
        .join(TEST_DIR)
        .join(test_name)
        .with_extension("sol");
    let chm_path = current_dir
        .join(TEST_DIR)
        .join(test_name)
        .with_extension("chm");
    let chp_path = current_dir
        .join(TEST_DIR)
        .join(test_name)
        .with_extension("chp");

    // Cleanup before running
    let _ = fs::remove_file(&log_path);
    let _ = fs::remove_file(&err_path);
    let _ = fs::remove_file(&output_path);
    let _ = fs::remove_file(&prf_path);
    let _ = fs::remove_file(&chm_path);
    let _ = fs::remove_file(&chp_path);

    let executable_path = current_dir.join("target/debug").join(EXECUTABLE_NAME);
    let mut cmd = Command::new(executable_path);
    cmd.arg(&cnf_path).arg(&output_path).arg(&prf_path);

    let output: Output = cmd.output()?;

    // Write output to log file and error to err file
    fs::File::create(&log_path)?.write_all(&output.stdout)?;
    fs::File::create(&err_path)?.write_all(&output.stderr)?;

    let expected_status = 0;
    if output.status.code().unwrap_or(1) != expected_status {
        return Err(format!(
            "Simplification failed: exit status '{}', expected '{}'",
            output.status.code().unwrap_or(1),
            expected_status
        )
        .into());
    }

    // Model checking if solution file exists
    if sol_path.exists() {
        let mut cmd = Command::new("checkmodel");
        cmd.arg(&output_path).arg(&sol_path);
        let output: Output = cmd.output()?;
        fs::File::create(&chm_path)?.write_all(&output.stdout)?;
        fs::File::create(&err_path)?.write_all(&output.stderr)?;

        if output.status.code().unwrap_or(1) != expected_status {
            return Err(format!(
                "Model checking failed: exit status '{}', expected '{}'",
                output.status.code().unwrap_or(1),
                expected_status
            )
            .into());
        }
    }

    // Proof checking
    let mut cmd = Command::new("checkproof");
    cmd.arg(&cnf_path).arg(&prf_path);
    let output: Output = cmd.output()?;
    fs::File::create(&chp_path)?.write_all(&output.stdout)?;
    fs::File::create(&err_path)?.write_all(&output.stderr)?;

    if output.status.code().unwrap_or(1) != expected_status {
        return Err(format!(
            "Proof checking failed: exit status '{}', expected '{}'",
            output.status.code().unwrap_or(1),
            expected_status
        )
        .into());
    }

    Ok(())
}

#[test]
fn run_all_tests() -> Result<(), Box<dyn std::error::Error>> {
    let test_files = fs::read_dir(TEST_DIR)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == CNF_EXT))
        .collect::<Vec<_>>();

    for test_file in test_files {
        let test_name = test_file.file_name().into_string().unwrap();
        run_test_case(&test_name.trim_end_matches(".cnf"))?;
    }

    Ok(())
}
