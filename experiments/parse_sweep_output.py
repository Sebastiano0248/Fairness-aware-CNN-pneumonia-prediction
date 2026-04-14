"""
Script para parsear sweep_bias_accuracy.out y generar un JSON estructurado.
"""

import re
import json

def parse_sweep_output(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Dividir por bloques de experimento
    blocks = re.split(r"={60}\n\[\d+/\d+\]", text)
    headers = re.findall(r"={60}\n\[(\d+)/(\d+)\]\s+alpha=([\d.]+),\s*ls=(\d+)", text)

    experiments = []

    for header, block in zip(headers, blocks[1:]):
        exp_num, total, alpha_h, ls_h = header

        exp = {
            "experiment": int(exp_num),
            "total_experiments": int(total),
            "alpha": float(alpha_h),
            "layer_size": int(ls_h),
        }

        # Config
        cfg = re.search(
            r"Config: alpha=([\d.]+), layer_size=(\d+), batch_size=(\d+), lr=([\d.]+)",
            block,
        )
        if cfg:
            exp["config"] = {
                "alpha": float(cfg.group(1)),
                "layer_size": int(cfg.group(2)),
                "batch_size": int(cfg.group(3)),
                "lr": float(cfg.group(4)),
            }

        # Data
        data_m = re.search(r"Data: (\d+) train / (\d+) test samples", block)
        if data_m:
            exp["data"] = {
                "train_samples": int(data_m.group(1)),
                "test_samples": int(data_m.group(2)),
            }

        # Results dir
        rd = re.search(r"Results dir: (.+)", block)
        if rd:
            exp["results_dir"] = rd.group(1).strip()

        # Iteraciones
        iter_blocks = re.split(r"Iteración (\d+)/(\d+)", block)
        iterations = []

        # iter_blocks[0] es texto antes de la primera iteración
        # luego van en trios: iter_num, total_iter, contenido
        for i in range(1, len(iter_blocks), 3):
            iter_num = int(iter_blocks[i])
            iter_total = int(iter_blocks[i + 1])
            content = iter_blocks[i + 2] if i + 2 < len(iter_blocks) else ""

            iteration = {
                "iteration": iter_num,
                "total_iterations": iter_total,
                "epochs": [],
                "metrics": {},
                "gender_prediction_auc": {},
            }

            # Epochs: parsear pares de líneas Loss + Epoch
            loss_lines = re.findall(
                r"Loss: ([\d.]+) \+ ([\d.]+)\*fairness penalty: ([\d.]+)"
                r" # Loss gradient norms: ([\d.]+)"
                r" # CE-Loss grad\. norms: ([\d.]+)",
                content,
            )
            epoch_lines = re.findall(
                r"Epoch (\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+)",
                content,
            )

            for loss_info, epoch_info in zip(loss_lines, epoch_lines):
                epoch_data = {
                    "epoch": int(epoch_info[0]),
                    "train_loss": float(epoch_info[1]),
                    "val_loss": float(epoch_info[2]),
                    "loss": float(loss_info[0]),
                    "alpha_multiplier": float(loss_info[1]),
                    "fairness_penalty": float(loss_info[2]),
                    "loss_gradient_norm": float(loss_info[3]),
                    "ce_loss_gradient_norm": float(loss_info[4]),
                }
                iteration["epochs"].append(epoch_data)

            # Métricas de accuracy/AUC
            for group in ["All", "Male", "Female"]:
                m = re.search(
                    rf"{group} - Accuracy: ([\d.]+), AUC: ([\d.]+)", content
                )
                if m:
                    iteration["metrics"][group.lower()] = {
                        "accuracy": float(m.group(1)),
                        "auc": float(m.group(2)),
                    }

            # Gender prediction AUC por capa
            layer_aucs = re.findall(
                r"Layer (\d+) - Gender prediction AUC: ([\d.]+)", content
            )
            for layer_num, auc_val in layer_aucs:
                iteration["gender_prediction_auc"][f"layer_{layer_num}"] = float(
                    auc_val
                )

            iterations.append(iteration)

        exp["iterations"] = iterations

        # Resultado final del experimento
        final = re.search(
            r"=> Accuracy: ([\d.]+), \|TPR gap\|: ([\d.]+)", block
        )
        if final:
            exp["final_accuracy"] = float(final.group(1))
            exp["final_tpr_gap"] = float(final.group(2))

        experiments.append(exp)

    return experiments


if __name__ == "__main__":
    input_file = "sweep_bias_accuracy.out"
    output_file = "sweep_bias_accuracy.json"

    data = parse_sweep_output(input_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Parseados {len(data)} experimentos → {output_file}")
