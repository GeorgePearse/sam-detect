//! Benchmark suite for inference performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array4;

fn benchmark_array_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_operations");

    for size in [224, 512, 1024].iter() {
        let array = black_box(Array4::<f32>::zeros((1, 3, *size, *size)));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    let _sum: f32 = array.iter().sum();
                    _sum
                })
            },
        );
    }

    group.finish();
}

fn benchmark_normalization(c: &mut Criterion) {
    let embedding = black_box(vec![3.0; 512]);

    c.bench_function("normalize_embedding_512d", |b| {
        b.iter(|| {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm == 0.0 {
                embedding.clone()
            } else {
                embedding.iter().map(|x| x / norm).collect::<Vec<_>>()
            }
        })
    });
}

criterion_group!(benches, benchmark_array_operations, benchmark_normalization);
criterion_main!(benches);
