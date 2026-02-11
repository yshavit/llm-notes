use simpllm_core::cputensor::CpuBackend;
use simpllm_core::run::run_main;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_main::<CpuBackend>()
}
