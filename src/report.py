"""Create aggregate figures and a local error-analysis report from real OOF."""
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "feedback-prize-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import CLASSES, LABELS


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def discourse_positions(gold, docs):
    """Aggregate gold span starts; each annotation contributes once."""
    word_counts = docs.set_index("id").words
    positions = gold.predictionstring.str.split().str[0].astype(int) / gold.id.map(word_counts)
    if positions.isna().any() or not positions.between(0, 1, inclusive="left").all():
        raise ValueError("Gold span starts must lie inside the corresponding essay")
    bins = np.minimum((positions * 10).astype(int), 9)
    counts = pd.crosstab(gold.discourse_type, bins).reindex(index=CLASSES, columns=range(10), fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    rows = []
    for label in CLASSES:
        values = positions.loc[gold.discourse_type == label]
        rows.append({"class": label, "span_count": len(values),
                     "median_relative_start": float(values.median()),
                     "first_20_percent_share": float((values < .2).mean()),
                     "last_20_percent_share": float((values >= .8).mean())})
    return pd.DataFrame(rows).set_index("class"), shares


def main():
    cfg = load("configs/baseline.json")
    root = Path(cfg["output_dir"])
    artifact = Path(cfg["artifacts_dir"])
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    results, audit = load(root / "metrics.json"), load(root / "data_audit.json")
    best = load(root / "selected_model.json")["experiment"]
    frame = pd.read_pickle(root / "private" / "candidates_oof.pkl")
    docs = pd.read_pickle(artifact / "documents.pkl")
    gold = pd.read_csv(Path(cfg["data_dir"]) / "train.csv")
    position_summary, position_shares = discourse_positions(gold, docs)
    position_summary.to_csv(root / "discourse_positions.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    counts = pd.Series(audit["class_counts"]).reindex(CLASSES)
    axes[0].barh(CLASSES, counts, color="#315b75")
    axes[0].set(title="Training annotations by discourse type", xlabel="Number of gold spans", ylabel="Discourse type")
    axes[0].invert_yaxis()
    axes[1].hist(docs.words, bins=50, color="#315b75")
    axes[1].set(title="Training essay lengths", xlabel="Whitespace-separated words per essay", ylabel="Number of essays")
    fig.suptitle("Feedback Prize 2021 · all 15,594 local training essays")
    fig.savefig(figures / "eda.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    im = ax.imshow(position_shares.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(len(CLASSES)):
        for j in range(10):
            share = position_shares.iloc[i, j]
            ax.text(j, i, f"{100 * share:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if share > .55 else "black")
    ax.set(xticks=range(10), xticklabels=[f"{i * 10}–{(i + 1) * 10}" for i in range(10)],
           yticks=range(len(CLASSES)), yticklabels=CLASSES,
           xlabel="Span start as percentage of essay word count (%)",
           ylabel="Gold discourse type",
           title="Where discourse spans begin · proportion within each class")
    fig.colorbar(im, ax=ax, label="Fraction of gold spans within class (0–1)")
    fig.suptitle(f"Source: train.csv and raw text word counts · {len(gold):,} training annotations", fontsize=10)
    fig.savefig(figures / "discourse_positions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    names = list(results)
    ax.bar(range(len(names)), [results[n]["fold_mean"] for n in names],
           yerr=[results[n]["fold_std"] for n in names], capsize=5, color="#315b75")
    ax.set(xticks=range(len(names)), xticklabels=[n.replace("_", "\n") for n in names],
           ylabel="Span macro F1 (0–1)", xlabel="Fixed model configuration",
           title="Mean span macro F1 ± sample standard deviation across 3 folds")
    fig.text(.01, .01, "Source: reports/fold_metrics.csv · same essay folds · seed 42", fontsize=8)
    fig.savefig(figures / "experiments.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    width = .25
    x = np.arange(len(CLASSES))
    for j, name in enumerate(["lexical", "lexical_position", "lexical_position_balanced"]):
        if name in results:
            ax.bar(x + (j - 1) * width, [results[name]["per_class"][c]["f1"] for c in CLASSES], width, label=name)
    ax.set(xticks=x, xticklabels=[c.replace(" ", "\n") for c in CLASSES],
           xlabel="Discourse type", ylabel="OOF span F1 (0–1)", title="Per-class span F1 · pooled 3-fold out-of-fold predictions")
    ax.legend(fontsize=8)
    fig.text(.01, .01, "Source: reports/metrics.json · all training essays predicted out of fold", fontsize=8)
    fig.savefig(figures / "class_f1.png", dpi=150)
    plt.close(fig)

    matrix = pd.read_csv(root / "candidate_confusion.csv", index_col=0).to_numpy()
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if normalized[i, j] > .55 else "black")
    ax.set(xticks=range(len(LABELS)), xticklabels=LABELS, yticks=range(len(LABELS)), yticklabels=LABELS,
           xlabel="Predicted candidate majority class", ylabel="True candidate majority class",
           title=f"Candidate confusion · {best} · OOF\nRows normalized; this is not the competition span metric")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.colorbar(im, ax=ax, label="Fraction of candidates within true class")
    fig.savefig(figures / "confusion.png", dpi=150)
    plt.close(fig)

    fold_rows = []
    for n, m in results.items():
        ci = m["bootstrap_ci95"]
        fold_rows.append([n, f'{m["fold_mean"]:.4f} ± {m["fold_std"]:.4f}', f'{m["macro_f1"]:.4f}',
                          f'[{ci[0]:.4f}; {ci[1]:.4f}]', f'{m["micro_f1"]:.4f}', f'{m["total_seconds"]:.1f}'])
    text = "# Измеренные результаты\n\nВсе числа получены локальным запуском; без GPU, внешних данных и leaderboard.\n\n"
    text += table(["Конфигурация", "Span macro F1, mean ± std", "Pooled OOF macro F1", "95% cluster-bootstrap CI OOF", "OOF micro F1", "Обучение + оценка 3 фолдов, с"], fold_rows)
    text += f'\n\nВыбор по среднему validation macro F1: **{best}**. Std — выборочный (ddof=1), не standard error. Интервал условен относительно split и выбора конфигураций.\n\n'
    text += "## Проверка трёх гипотез\n\n"
    deltas = load(root / "paired_comparisons.json")
    for key, d in deltas.items():
        ci = d["paired_bootstrap_ci95"]
        text += f'- `{key}`: Δ pooled OOF macro F1 = {d["delta_oof"]:+.4f}; парный 95% CI [{ci[0]:+.4f}; {ci[1]:+.4f}].\n'
    a, b = results["lexical_position_balanced"], results["lexical_position"]
    rare_a = np.mean([a["per_class"][c]["recall"] for c in ["Counterclaim", "Rebuttal"]])
    rare_b = np.mean([b["per_class"][c]["recall"] for c in ["Counterclaim", "Rebuttal"]])
    text += f'\nH3, средний recall Counterclaim/Rebuttal: {rare_b:.4f} → {rare_a:.4f}; Δ={rare_a - rare_b:+.4f}. Это описательное сравнение recall.\n\n'
    text += "## Метрики выбранной модели по классам\n\n"
    rows = []
    for c in CLASSES:
        m = results[best]["per_class"][c]
        rows.append([c, f'{m["precision"]:.4f}', f'{m["recall"]:.4f}', f'{m["f1"]:.4f}', m["tp"], m["fp"], m["fn"]])
    text += table(["Класс", "Precision", "Recall", "F1", "TP", "FP", "FN"], rows)
    text += "\n\n## Срезы длины эссе\n\n"
    slices = load(root / "length_slices.json")
    text += table(["Длина", "Эссе", "OOF span macro F1"], [[k, v["documents"], f'{v["macro_f1"]:.4f}'] for k, v in slices.items()])
    diag = load(root / "candidate_gold_diagnostic.json")
    text += f'\n\nДиагностика кандидатной сегментации: если присвоить каждому кандидату истинный доминирующий класс и применить тот же decoder, span macro F1 = {diag["macro_f1"]:.4f}. Это использует gold, не является deployable-моделью или строгой верхней границей, и не участвует в выборе модели.\n'
    text += f'\nВторичная метрика выбранной модели: candidate macro F1 (8 классов, включая O) = {results[best]["candidate_macro_f1"]:.4f}, accuracy = {results[best]["candidate_accuracy"]:.4f}. Она оценивает суррогатную задачу на автоматически выделенных кандидатах, а не качество извлечения аргументов.\n\n'
    text += "## Положение аргументов в эссе\n\n"
    text += "Начало gold-фрагмента делится на число исходных whitespace-слов в эссе. Каждый фрагмент учитывается один раз; строки графика нормированы отдельно для каждого класса.\n\n"
    text += table(["Класс", "Медиана начала, % длины эссе", "Начинаются в первых 20%, % фрагментов", "Начинаются в последних 20%, % фрагментов"],
                  [[label, f'{100 * row.median_relative_start:.1f}', f'{100 * row.first_20_percent_share:.1f}', f'{100 * row.last_20_percent_share:.1f}']
                   for label, row in position_summary.iterrows()])
    lead, conclusion = position_summary.loc["Lead"], position_summary.loc["Concluding Statement"]
    text += f'\n\nВ первых 20% эссе начинаются {100 * lead.first_20_percent_share:.1f}% фрагментов Lead; в последних 20% — {100 * conclusion.last_20_percent_share:.1f}% заключений. Медианы начала: {100 * lead.median_relative_start:.1f}% и {100 * conclusion.median_relative_start:.1f}% длины эссе соответственно. Это объясняет мотивацию H2; эффект признаков проверен отдельной CV-абляцией. Статистика описывает всю train-выборку и не используется для настройки модели.\n\n'
    text += "![Положение аргументов](figures/discourse_positions.png)\n\n"
    text += "## Графики\n\n![EDA](figures/eda.png)\n\nРазмеры классов и длины эссе показывают дисбаланс и разную сложность документов.\n\n![Эксперименты](figures/experiments.png)\n\nСравнение идёт на одинаковых фолдах; ошибки на графике — std между фолдами.\n\n![Классы](figures/class_f1.png)\n\nПоклассовый F1 позволяет отделить улучшение частых типов от качества редких.\n\n![Ошибки](figures/confusion.png)\n\nМатрица относится к кандидатной классификации; ошибки границ оценивает span F1.\n"
    (root / "RESULTS.md").write_text(text, encoding="utf-8")

    # Real examples remain local, since they include restricted student writing.
    errors = frame.loc[frame.target != frame.prediction].copy()
    errors["words"] = errors.end - errors.start
    pairs = errors.groupby(["target", "prediction"]).size().sort_values(ascending=False)
    pairs.rename("candidate_count").to_csv(root / "error_pairs.csv")
    examples = "# Три реальных OOF-ошибки (локальный файл, не публиковать)\n\n"
    examples += f"Модель: {best}. Это ошибки класса кандидата; ниже отдельно показана исходная span-разметка и предсказания.\n\n"
    predictions = pd.read_csv(root / "predictions" / f"{best}_oof.csv")
    selected = []
    for wanted in [("Claim", "Evidence"), ("Rebuttal", "Counterclaim"), ("Position", "Claim")]:
        subset = errors.loc[(errors.target == wanted[0]) & (errors.prediction == wanted[1]) & (errors.words.between(8, 55))]
        if len(subset):
            selected.append(subset.sort_values(["id", "start"]).iloc[0])
    for i, r in enumerate(selected, 1):
        examples += f'## {i}. {r.target} → {r.prediction}\n\nЭссе `{r.id}`, fold {r.fold}, слова [{r.start}, {r.end}) по исходному `.split()`.\n\n> {r.text.replace(chr(10), " ")}\n\n'
        for label, group in [("Gold", gold.loc[gold.id == r.id]), ("OOF prediction", predictions.loc[predictions.id == r.id])]:
            examples += f"{label}, фрагменты, пересекающие кандидата:\n\n"
            for span in group.itertuples(index=False):
                indices = list(map(int, str(span.predictionstring).split()))
                if set(indices) & set(range(r.start, r.end)):
                    klass = span.discourse_type if label == "Gold" else getattr(span, "_1")
                    examples += f"- {klass}: слова {min(indices)}–{max(indices)} ({len(indices)} слов).\n"
            examples += "\n"
        reason = {"Claim": "Короткое утверждение содержит фактическую лексику; без контекста модель воспринимает его как подтверждение. Сравнение границ показывает, как candidate decoder может расширять фрагмент.",
                  "Rebuttal": "И возражение, и ответ на него используют лексику несогласия. Для различения нужна связь с предыдущим аргументом, которой нет в локальном TF-IDF.",
                  "Position": "Позиция автора и локальный тезис похожи по формулировкам. Одна позиция в эссе не задаётся ограничением декодера, поэтому локальный классификатор может спутать их."}.get(r.target, "Ошибка локального контекста.")
        examples += f"Интерпретация (гипотеза, а не установленная причина): {reason}\n\n"
    (root / "private" / "error_examples.md").write_text(examples, encoding="utf-8")
    print(f"Created reports/RESULTS.md, 5 figures, position statistics and {len(selected)} private OOF error examples")


if __name__ == "__main__":
    main()
