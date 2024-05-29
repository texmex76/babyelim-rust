use anyhow::{Context, Result};
use assert_cmd::Command;
use std::fs;
use std::io::Write;
use std::process::Output;
use test_case::test_case;

// Path constants for the tests directory and test files
const TEST_DIR: &str = "tests/test_cases";
const CNF_EXT: &str = "cnf";
const EXECUTABLE_NAME: &str = "babyelim-rust";

fn run_test_case(test_name: &str) -> Result<()> {
    let current_dir = std::env::current_dir().context("Failed to get current directory")?;
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
    if log_path.exists() {
        fs::remove_file(&log_path).context("Failed to remove log file")?;
    }
    if err_path.exists() {
        fs::remove_file(&err_path).context("Failed to remove error file")?;
    }
    if output_path.exists() {
        fs::remove_file(&output_path).context("Failed to remove output file")?;
    }
    if prf_path.exists() {
        fs::remove_file(&prf_path).context("Failed to remove proof file")?;
    }
    if chm_path.exists() {
        fs::remove_file(&chm_path).context("Failed to remove model check file")?;
    }
    if chp_path.exists() {
        fs::remove_file(&chp_path).context("Failed to remove proof check file")?;
    }

    let executable_path = current_dir.join("target/debug").join(EXECUTABLE_NAME);
    let mut cmd = Command::new(executable_path);
    cmd.arg(&cnf_path).arg(&prf_path).arg(&output_path);

    let output: Output = cmd.output().context("Failed to execute command")?;

    // Write output to log file and error to err file
    fs::File::create(&log_path)?
        .write_all(&output.stdout)
        .context("Failed to write to log file")?;
    fs::File::create(&err_path)?
        .write_all(&output.stderr)
        .context("Failed to write to error file")?;

    let expected_status = 0;
    if output.status.code().unwrap_or(1) != expected_status {
        return Err(anyhow::anyhow!(
            "Simplification failed: exit status '{}', expected '{}'",
            output.status.code().unwrap_or(1),
            expected_status
        ));
    }

    // Model checking if solution file exists
    if sol_path.exists() {
        let mut cmd = Command::new("./checkmodel");
        cmd.arg(&output_path).arg(&sol_path);
        let output: Output = cmd
            .output()
            .context("Failed to execute model checking command")?;
        fs::File::create(&chm_path)?
            .write_all(&output.stdout)
            .context("Failed to write to model check file")?;
        fs::File::create(&err_path)?
            .write_all(&output.stderr)
            .context("Failed to write to error file")?;

        if output.status.code().unwrap_or(1) != expected_status {
            return Err(anyhow::anyhow!(
                "Model checking failed: exit status '{}', expected '{}'",
                output.status.code().unwrap_or(1),
                expected_status
            ));
        }
    }

    // Proof checking
    let mut cmd = Command::new("./checkproof");
    cmd.arg(&cnf_path).arg(&prf_path);
    let output: Output = cmd
        .output()
        .context("Failed to execute proof checking command")?;
    fs::File::create(&chp_path)?
        .write_all(&output.stdout)
        .context("Failed to write to proof check file")?;
    fs::File::create(&err_path)?
        .write_all(&output.stderr)
        .context("Failed to write to error file")?;

    if output.status.code().unwrap_or(1) != expected_status {
        return Err(anyhow::anyhow!(
            "Proof checking failed: exit status '{}', expected '{}'",
            output.status.code().unwrap_or(1),
            expected_status
        ));
    }

    Ok(())
}

#[test_case("false")]
#[test_case("prime121")]
#[test_case("prime1369")]
#[test_case("prime1681")]
#[test_case("prime169")]
#[test_case("prime1849")]
#[test_case("prime2209")]
#[test_case("prime25")]
#[test_case("prime289")]
#[test_case("prime361")]
#[test_case("prime49")]
#[test_case("prime4")]
#[test_case("prime529")]
#[test_case("prime841")]
#[test_case("prime961")]
#[test_case("prime9")]
#[test_case("simp")]
#[test_case("sqrt10201")]
#[test_case("sqrt1042441")]
#[test_case("sqrt10609")]
#[test_case("sqrt11449")]
#[test_case("sqrt11881")]
#[test_case("sqrt12769")]
#[test_case("sqrt16129")]
#[test_case("sqrt259081")]
#[test_case("sqrt2809")]
#[test_case("sqrt3481")]
#[test_case("sqrt3721")]
#[test_case("sqrt4489")]
#[test_case("sqrt5041")]
#[test_case("sqrt5329")]
#[test_case("sqrt6241")]
#[test_case("sqrt63001")]
#[test_case("sqrt6889")]
#[test_case("sqrt7921")]
#[test_case("sqrt9409")]
#[test_case("trivial")]
#[test_case("true")]
#[test_case("unit1")]
#[test_case("unit2")]
#[test_case("unit3")]
#[test_case("unit4")]
#[test_case("unit5")]
#[test_case("unit6")]
fn test_cases(test_name: &str) {
    run_test_case(test_name).unwrap();
}
