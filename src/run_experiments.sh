#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash src/run_experiments.sh STAGE DATASET_ROOT MODEL_ROOT [OUTPUT_ROOT]
  bash src/run_experiments.sh mia [--model MODEL] DATASET_ROOT MODEL_ROOT [OUTPUT_ROOT]
  bash src/run_experiments.sh plots [RESULTS_ROOT [PLOT_OUTPUT_ROOT]]

Activate the Python environment described in README.md first.
DATASET_ROOT contains <dataset>/data/{train,val}/<image_id>.jpg and 
<dataset>/metadata/meta_<dataset>.jsonl. MODEL_ROOT contains <model>/fp16/.
OUTPUT_ROOT defaults to exp_data in the repository. 
Explicit relative paths are resolved from the directory you run the command in.

Stages:
  all         Run data preparation, feature extraction, RQ1-RQ6, similarity
              measurements, MIA ablation, and figures and tables. Use a fresh
              output root for the end-to-end timing measurements.
  prepare     Generate the source manifest, SCNS pairs, and CLIP-KNN hold-outs.
  features    Extract MIA features for all three models. Run `prepare` first.
  rq1         Run seven measured-distance levels and draw the vulnerability of 
              standard baseline DI under realistic distribution shifts.
  rq2, rq5    Run the shared calibration statistics and draw the selected grid.
              Both stages use the same 1,500 candidate-pair statistics.
  rq3         Run calibrated DI at K=400 and draw the calibration comparison.
  rq4         Measure standard DI runtime and summarize costs. Requires timing
              files from a full `features` run and a fresh `rq3` run.
  rq6         Run the eight SD1.5 dataset sizes and draw the comparison.
  similarity  Measure semantic and textural distances. Run `prepare` first.
  ablation    Run and export the MIA ablation. Run `prepare`, `features`, and
              `rq3` first.
  plots       Generate figures and tables from completed RQ1, RQ2/RQ5, RQ3,
              RQ6, and `similarity` inputs. Specify a separate PLOT_OUTPUT_ROOT
              if the source root already contains RQ3 or RQ6 plots.
  mia         Run the standalone StructuralMI baseline. Run `prepare` first.
              Choose `stable-diffusion-v1p5`, `commoncanvas-s-c`, or `kandinsky-3`
              with `--model`. The default is `stable-diffusion-v1p5`.

For individual RQ stages, run `prepare` and `features` first, then reuse their
OUTPUT_ROOT and the same source paths. RQ3 and RQ6 require fresh figure
outputs. Use the same experiment root for all inputs to each stage.
For example:
  bash src/run_experiments.sh all /data/images /data/models /data/experiment
  bash src/run_experiments.sh rq1 /data/images /data/models /data/experiment
  bash src/run_experiments.sh mia --model commoncanvas-s-c /data/images /data/models /data/experiment
EOF
}

stage=${1:-}
case "$stage" in
  -h|--help) usage; exit 0 ;;
  all|prepare|features|rq1|rq2|rq3|rq4|rq5|rq6|similarity|ablation|plots|mia) shift ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "$stage" == mia && "${1:-}" == --model ]]; then
  if (( $# < 4 )); then usage >&2; exit 2; fi
  mia_model=$2
  shift 2
fi

absolute_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *)  printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

if [[ "$stage" == plots ]]; then
  if (( $# > 2 )); then usage >&2; exit 2; fi
  output_arg=${1:-}
  if [[ -n "${2:-}" ]]; then plot_output=$(absolute_path "$2"); fi
else
  if (( $# < 2 || $# > 3 )); then usage >&2; exit 2; fi
  dataset_root=$(absolute_path "$1")
  model_root=$(absolute_path "$2")
  output_arg=${3:-}
fi

output=exp_data
if [[ -n "$output_arg" ]]; then output=$(absolute_path "$output_arg"); fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
seed=12345
models=(stable-diffusion-v1p5 commoncanvas-s-c kandinsky-3)
data_scripts=src/exp_utils/dataset_process_util
plot_scripts=src/exp_utils/plot_util
sd=stable-diffusion-v1p5
laion=laion_aesthetics_v2_5p0
cc=commoncatalog_cc_by_sa
metadata="$output/experiments/shared_res/datasets"
features="$output/experiments/shared_res/mia_features/offline_features"
rq3_results="$output/experiments/RQ3_calibration_performance"
analysis="$output/plot_analyze"

if [[ "$stage" != plots ]]; then
  experiment_args=(
    --base_model_id "$model_root" --local_dataset_path "$dataset_root"
    --output_root "$output" --global_seed "$seed"
    --dataset_total_num 1000 --split_times 100
    --sample_manifest "$metadata/sample_manifest.json"
  )
fi

prepare() {
  python "$data_scripts/prepare_experiment_data.py" --local_dataset_path "$dataset_root" \
    --output_root "$output" --global_seed "$seed"
  python "$data_scripts/generate_hold_out_datasets.py" "${experiment_args[@]}" --model_name "$sd"
  python "$data_scripts/emp_null_sample.py" "${experiment_args[@]}" --model_name "$sd" \
    --experiment rq2_rq5 --num_pairs 1500
  python "$data_scripts/clip_knn.py" "${experiment_args[@]}" --model_name "$sd" \
    --used_datasets "$laion" "D_hold_out_$laion"
  python "$data_scripts/clip_knn.py" "${experiment_args[@]}" --model_name commoncanvas-s-c \
    --used_datasets "$cc" "D_hold_out_$cc"
}

extract_features() {
  python src/extract_features.py "${experiment_args[@]}" --model_name "$1"
}

# The main comparison and the RQ2/RQ5 grids use the same three types of tests.
test_pairs() {
  local model=$1 target=$2 count=$3
  shift 3
  python src/main_ttest.py "${experiment_args[@]}" --model_name "$model" \
    "$@" --used_datasets D_base_1 D_base_2 --k_base "$count"
  python src/main_ttest.py "${experiment_args[@]}" --model_name "$model" \
    "$@" --used_datasets D_hold_out_1 D_hold_out_2
  python src/main_ttest.py "${experiment_args[@]}" --model_name "$model" \
    "$@" --used_datasets "$target" "D_hold_out_$target"
}

rq1() {
  python "$plot_scripts/rq1_run.py" "${experiment_args[@]}" --model_name "$1" --experiment rq1 \
    --rq1_steps 6 --rq1_candidates 100 --rq1_shift_min 0.06 --rq1_shift_max "$2" \
    --rq1_max_mix 0.6 --rq1_nocaps_max 0.3 --feature_root "$features"
}

rq6() {
  local pair left right
  for pair in "D_base_1 D_base_2" "D_hold_out_1 D_hold_out_2" "$laion D_hold_out_$laion"; do
    read -r left right <<< "$pair"
    python "$plot_scripts/rq6_run_dataset_sizes.py" "${experiment_args[@]}" \
      --model_name "$sd" --used_datasets "$left" "$right" --k_base 500 \
      --dataset_sizes 10 20 50 100 200 300 400 500
  done
}

baseline_time() {
  python src/baseline_ttest.py "${experiment_args[@]}" --model_name "$sd" \
    --used_datasets "$laion" "D_hold_out_$laion"
}

similarity() {
  local pair model left right split_left split_right measure
  for pair in \
    "$sd coco_2017_val coco_2017_val val val" \
    "$sd laion2b_multi_translated coco_2017_val val val" \
    "$sd nocaps coco_2017_val val val" \
    "$sd $laion D_hold_out_$laion train val" \
    "commoncanvas-s-c $cc D_hold_out_$cc train val" \
    "$sd D_hold_out_1 D_hold_out_2 val val"; do
    read -r model left right split_left split_right <<< "$pair"
    for measure in seman_bound texture_bound; do
      python "$data_scripts/$measure.py" "${experiment_args[@]}" \
        --model_name "$model" --used_datasets "$left" "$right" --splits "$split_left" "$split_right"
    done
  done
}

calibrate() {
  local model
  for model in "${models[@]}"; do
    python src/calculate_threshold_fpr.py "${experiment_args[@]}" --model_name "$model" --k_base 400
  done
}

ablation() {
  python src/exp_utils/mi_util/mia_ablation/run_mia_ablation_all.py "${experiment_args[@]}" \
    --models "${models[@]}" --feature_root "$features" --metadata_root "$metadata" \
    --main_results "$rq3_results" --num_pairs 500 --k_base 400 --workers 8
}

plots() {
  python "$plot_scripts/plot_all.py" --results "$output" --output "${plot_output:-$output}" --global_seed "$seed"
}

rq4_summary() {
  local target_stats="time_stats_${laion}_D_hold_out_${laion}.json"
  python "$plot_scripts/rq4_analyze_time_cost.py" \
    --offline "$features/time_stats_$sd.json" \
    --standard "$output/experiments/RQ1_standard_di_vulnerability/cdi_baseline/$sd/$target_stats" \
    --target "$rq3_results/$sd/ttest/$target_stats" \
    --calibration "$rq3_results/$sd/ttest/time_stats_D_base_1_D_base_2.json" \
    --output "$analysis/RQ4_time_cost/timing_summary.json"
}

if [[ "$stage" == all || "$stage" == prepare ]]; then prepare; fi
case "$stage" in
  all|features|rq1|rq2|rq3|rq5)
    for model in "${models[@]}"; do
      if [[ "$model" == commoncanvas-s-c ]]; then
        target=$cc
        rq1_shift_max=0.14
      else
        target=$laion
        rq1_shift_max=0.22
      fi

      case "$stage" in all|features) extract_features "$model" ;; esac
      case "$stage" in all|rq3) test_pairs "$model" "$target" 500 ;; esac
      case "$stage" in all|rq2|rq5) test_pairs "$model" "$target" 1500 --experiment rq2_rq5 ;; esac
      case "$stage" in all|rq1) rq1 "$model" "$rq1_shift_max" ;; esac
    done
    ;;
esac

case "$stage" in
  all)
    rq6
    baseline_time
    similarity
    calibrate
    ablation
    plots
    rq4_summary
    ;;
  rq1)
    python "$plot_scripts/rq1_plot_vulnerability.py" --results "$output" \
      --output "$analysis/RQ1_standard_di_vulnerability"
    ;;
  rq2|rq5)
    python "$plot_scripts/rq2_rq5_hyperparameter_analysis.py" --results "$output" \
      --output "$output" --case "$stage" --global_seed "$seed"
    ;;
  rq3)
    calibrate
    python "$plot_scripts/rq3_plot_calibration.py" --results "$output" \
      --output "$analysis/RQ3_calibration_performance"
    ;;
  rq4)
    baseline_time
    rq4_summary
    ;;
  rq6)
    rq6
    csv="$analysis/RQ6_dataset_size/stable-diffusion-v1-5.csv"
    python "$plot_scripts/rq6_analyze_dataset_sizes.py" --results "$output" --output "$csv"
    python "$plot_scripts/rq6_plot_dataset_sizes.py" --csv "$csv" \
      --output "$analysis/RQ6_dataset_size/rq6_stable-diffusion-v1-5.pdf"
    ;;
  similarity) similarity ;;
  ablation) ablation ;;
  plots) plots ;;
  mia)
    model=${mia_model:-$sd}
    target=$laion
    if [[ "$model" == commoncanvas-s-c ]]; then target=$cc; fi
    python src/exp_utils/mi_util/baseline_mi_main.py "${experiment_args[@]}" --model_name "$model" \
      --used_datasets "$target" "D_hold_out_$target"
    ;;
esac
