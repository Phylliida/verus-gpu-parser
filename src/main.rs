///  verus-gpu-transpile: Parse a Verus #[gpu_kernel] function and emit WGSL.
///
///  Usage: verus-gpu-transpile <input.rs> [-o <output.wgsl>]

mod types;
mod emit;
mod parse;

use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: verus-gpu-transpile <input.rs> [-o <output.wgsl>]");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = args.iter().position(|a| a == "-o")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    let source = fs::read_to_string(input_path)
        .unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", input_path, e);
            std::process::exit(1);
        });

    let kernel = match parse::parse_gpu_kernel(&source) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };

    let wgsl = emit::emit_kernel(&kernel);

    if let Some(out) = output_path {
        fs::write(out, &wgsl).unwrap_or_else(|e| {
            eprintln!("Error writing {}: {}", out, e);
            std::process::exit(1);
        });
        eprintln!("Wrote {} bytes to {}", wgsl.len(), out);
    } else {
        println!("{}", wgsl);
    }
}
