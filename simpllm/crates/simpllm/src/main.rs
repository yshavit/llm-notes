use simpllm::cputensor::CpuBackend;
use simpllm::run::run_main;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_main::<CpuBackend>()
}
