mod run;

use simpllm::run_main;
use simpllm_core::cputensor::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_main::<CpuBackend>()
}
