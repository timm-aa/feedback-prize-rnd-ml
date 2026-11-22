"""Build the HW2 report and figures from measured experiment artifacts."""
import json
import os
from pathlib import Path
import tempfile

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


def interval(ci):
    return f"[{ci[0]:+.4f}; {ci[1]:+.4f}]"


def main():
    cfg = load("configs/hw2.json")
    root = Path(cfg["output_dir"])
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    results = load(root / "metrics.json")
    comparisons = load(root / "paired_comparisons.json")
    selected = load(root / "selected_model.json")["name"]
    meta = load(root / "run_metadata.json")
    checks = load(root / "verification.json")
    split = pd.read_csv(root / "splits/essay_folds.csv")
    gold = pd.read_csv(Path(cfg["data_dir"]) / "train.csv")
    class_counts = pd.crosstab(gold.discourse_type, gold.id.map(split.set_index("id").fold)).reindex(CLASSES)
    if not (class_counts > 0).all().all():
        raise ValueError("Every evaluation fold must contain each discourse class")
    class_counts.to_csv(root / "fold_class_counts.csv")
    pd.crosstab(split.stratum, split.fold).to_csv(root / "stratum_counts.csv")
    names = [e["name"] for e in cfg["experiments"]]
    rows = []
    for e in cfg["experiments"]:
        m = results[e["name"]]
        rows.append({"experiment": e["name"], "hypothesis": e["hypothesis"], "context": e["context"],
                     "selective_merge": e["selective_merge"], "seed": cfg["seed"],
                     "fold_0": m["fold_values"][0], "fold_1": m["fold_values"][1], "fold_2": m["fold_values"][2],
                     "fold_mean": m["fold_mean"], "fold_std": m["fold_std"], "oof_macro_f1": m["macro_f1"],
                     "oof_micro_f1": m["micro_f1"], "ci95_low": m["bootstrap_ci95"][0],
                     "ci95_high": m["bootstrap_ci95"][1], "delta_from_baseline": m["macro_f1"] - results["baseline"]["macro_f1"]})
    pd.DataFrame(rows).to_csv(root / "experiments.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(x, [results[n]["fold_mean"] for n in names], color="#315b75",
           yerr=[results[n]["fold_std"] for n in names], capsize=5)
    for fold in range(3):
        ax.scatter(x + (fold - 1) * .08, [results[n]["fold_values"][fold] for n in names],
                   color="#1b2730", s=14, zorder=3)
    ax.set(xticks=x, xticklabels=[n.replace("_", "\n") for n in names],
           xlabel="Experiment (same 3 essay folds)", ylabel="Span macro F1 (0–1)",
           title="HW2: mean validation macro F1 ± fold standard deviation")
    fig.get_layout_engine().set(rect=(0, .065, 1, .935))
    fig.text(.01, .01, "Source: reports/hw2/fold_metrics.csv · seed 42 · dots = individual folds", fontsize=8)
    fig.savefig(figures / "experiments.png", dpi=150)
    plt.close(fig)
    effects = [("H1: context vs baseline", "context_only_minus_baseline"),
               ("H2: decoder vs baseline", "decoder_only_minus_baseline"),
               ("Full − without context", "context_decoder_minus_decoder_only"),
               ("Full − without decoder", "context_decoder_minus_context_only")]
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    for i, (label, key) in enumerate(effects):
        d = comparisons[key]
        low, high = d["paired_bootstrap_ci95"]
        ax.plot([low, high], [i, i], color="#315b75", linewidth=3)
        ax.scatter([d["delta_oof"]], [i], color="#1b2730", zorder=3)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(yticks=range(len(effects)), yticklabels=[e[0] for e in effects],
           xlabel="Difference in pooled OOF span macro F1 (absolute units)", ylabel="Fixed comparison",
           title="Paired effects and ablations · 95% cluster-bootstrap intervals")
    ax.invert_yaxis()
    fig.get_layout_engine().set(rect=(0, .065, 1, .935))
    fig.text(.01, .01, "Source: paired_comparisons.json · 1,000 resamples of essay groups · fixed OOF predictions", fontsize=8)
    fig.savefig(figures / "ablations.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for j, name in enumerate(names):
        ax.bar(np.arange(7) + (j - 1.5) * .2,
               [results[name]["per_class"][c]["f1"] for c in CLASSES], .2, label=name)
    ax.set(xticks=range(7), xticklabels=[c.replace(" ", "\n") for c in CLASSES],
           xlabel="Discourse type", ylabel="Pooled OOF span F1 (0–1)", title="Effects by discourse class")
    ax.legend(fontsize=8)
    fig.get_layout_engine().set(rect=(0, .065, 1, .935))
    fig.text(.01, .01, "Source: reports/hw2/metrics.json · identical OOF essays", fontsize=8)
    fig.savefig(figures / "class_f1.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for ax, name in zip(axes, ["baseline", "context"]):
        matrix = pd.read_csv(root / f"candidate_confusion_{name}.csv", index_col=0).to_numpy()
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(8):
            for j in range(8):
                value = normalized[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > .55 else "black")
        ax.set(xticks=range(8), xticklabels=LABELS, yticks=range(8), yticklabels=LABELS,
               xlabel="Predicted candidate class", ylabel="True majority candidate class", title=name)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(im, ax=axes, label="Fraction within true class (0–1)", shrink=.7)
    fig.suptitle("Candidate confusion · rows normalized · classifier diagnostic, not span matching")
    fig.savefig(figures / "confusion.png", dpi=150)
    plt.close(fig)

    text = "# ДЗ2. Feedback Prize: улучшения, трекинг и абляции\n\n"
    text += "Продолжение baseline из ДЗ1 на тех же 15 594 эссе и 144 293 gold-фрагментах. "
    text += f'Выбранная по среднему validation macro F1 конфигурация — **{selected}**: **{results[selected]["fold_mean"]:.4f} ± {results[selected]["fold_std"]:.4f}** по трём фолдам.\n\n'
    text += "## 1. Два изменения и гипотезы\n\n"
    text += "План и критерии сравнения записаны до просмотра результатов в [HW2_PLAN.md](docs/HW2_PLAN.md). Опорная модель — `lexical_position_balanced` из ДЗ1, переобученная в этом запуске.\n\n"
    text += "**H1 — контекст.** Низкие F1 Claim и Rebuttal в ДЗ1 связываем с отсутствием контекста. Добавлены два отдельных блока TF-IDF предыдущего и следующего кандидата, каждый с весом 0,35. Они используют тот же train-only словарь и IDF, что локальный текст. За границу эссе контекст не переходит; метки соседей не используются. Ожидаются рост macro span F1 и среднего recall Counterclaim/Rebuttal.\n\n"
    text += "**H2 — границы коротких аргументов.** В ДЗ1 Claim в среднем занимает 17,4 слова, Evidence — 77,4. Новый decoder объединяет соседние одинаковые предсказания только для Lead, Evidence и Concluding Statement. Для Claim, Position, Counterclaim и Rebuttal сохраняет кандидатные границы. Ожидаются рост recall Claim и общего macro F1; риск — разделение длинного аргумента на лишние predictions. Список классов и вес контекста заранее фиксированы.\n\n"
    text += "## 2. История экспериментов\n\n"
    text += table(["Эксперимент", "Контекст", "Selective decoder", "Seed", "Macro F1, mean ± std", "OOF macro F1", "Δ к baseline"],
                  [[r["experiment"], "да" if r["context"] else "нет", "да" if r["selective_merge"] else "нет", r["seed"],
                    f'{r["fold_mean"]:.4f} ± {r["fold_std"]:.4f}', f'{r["oof_macro_f1"]:.4f}', f'{r["delta_from_baseline"]:+.4f}'] for r in rows])
    text += "\n\nАртефакты трекинга:\n\n- [experiments.csv](reports/hw2/experiments.csv): гипотеза, компоненты, seed, все три фолда, mean/std, OOF и CI.\n- [fold_metrics.csv](reports/hw2/fold_metrics.csv): 12 оценок, размеры train/val, словарь, число признаков, эпохи, время, предупреждения, hashes.\n- [events.jsonl](reports/hw2/events.jsonl): события с UTC-временем и идентификатором общего обучения.\n- [config_resolved.json](reports/hw2/config_resolved.json) и [run_metadata.json](reports/hw2/run_metadata.json): все параметры, версии, hashes данных, split и кода.\n\n"
    text += f'Всего {checks["fits"]} обучений классификатора и {checks["experiment_fold_records"]} оценок: варианты с разными decoder используют одинаковые кандидатные предсказания. Это изолирует вклад decoder и не удваивает обучение. Полный запуск занял {meta["elapsed_seconds"]:.1f} с. Трекинг локальный, CSV + JSONL, без внешнего сервиса.\n\n'
    text += "![Сравнение экспериментов](reports/hw2/figures/experiments.png)\n\nСтолбцы — среднее по фолдам, полосы — выборочный std (ddof=1), точки — отдельные фолды.\n\n"
    text += "## 3. Абляции и эффекты\n\n"
    effect_rows = []
    for label, key in effects:
        d = comparisons[key]
        verdict = {"positive": "положительный", "negative": "отрицательный", "inconclusive": "CI включает ноль"}[d["interpretation"]]
        effect_rows.append([label, f'{d["delta_oof"]:+.4f}', interval(d["paired_bootstrap_ci95"]), verdict])
    text += table(["Сравнение", "Δ OOF macro F1", "Парный 95% CI", "Эффект"], effect_rows)
    text += "\n\nАбляция 1: из `context_decoder` удалён контекст → `decoder_only`. Абляция 2: из `context_decoder` удалён selective decoder → `context_only`. В каждой меняется ровно один компонент; остальные параметры, split и предсказания классификатора (для decoder) зафиксированы.\n\n"
    for label, key in effects:
        d = comparisons[key]
        conclusion = {"positive": "Компонент повышает macro F1 на этом протоколе.",
                      "negative": "Компонент ухудшает macro F1; как улучшение качества не принимается.",
                      "inconclusive": "Разницу не удаётся отделить от вариативности ресемплинга на этом split."}[d["interpretation"]]
        text += f'- **{label}:** {conclusion}\n'
    interaction = comparisons["interaction"]
    text += f'\nВзаимодействие компонентов (full − context_only − decoder_only + baseline): {interaction["delta_oof"]:+.4f}, 95% CI {interval(interaction["paired_bootstrap_ci95"])}. Нулевая величина соответствует аддитивности по шкале macro F1; ненулевая показывает зависимость эффекта от второго компонента, но не устанавливает механизм.\n\n'
    text += "![Абляции](reports/hw2/figures/ablations.png)\n\nИнтервалы получены парным ресемплингом одних и тех же групп эссе; вертикальная линия — нулевой эффект.\n\n"
    text += "## 4. Поклассовый анализ и ошибки\n\n"
    text += table(["Тип", *names], [[c, *[f'{results[n]["per_class"][c]["f1"]:.4f}' for n in names]] for c in CLASSES])
    base_recall = np.mean([results["baseline"]["per_class"][c]["recall"] for c in ["Counterclaim", "Rebuttal"]])
    ctx_recall = np.mean([results["context_only"]["per_class"][c]["recall"] for c in ["Counterclaim", "Rebuttal"]])
    text += f'\n\nH1, средний recall Counterclaim/Rebuttal: {base_recall:.4f} → {ctx_recall:.4f}. H2, recall Claim: {results["baseline"]["per_class"]["Claim"]["recall"]:.4f} → {results["decoder_only"]["per_class"]["Claim"]["recall"]:.4f}. Эти recall-сравнения описательны, отдельные интервалы для них не строились.\n\n'
    text += table(["Эксперимент", "OOF micro F1", "Candidate macro F1, 8 классов", "Предсказанных spans"],
                  [[n, f'{results[n]["micro_f1"]:.4f}', f'{results[n]["candidate_macro_f1"]:.4f}', results[n]["prediction_spans"]] for n in names])
    text += "\n\nCandidate F1 одинаков для пар с одним классификатором: decoder влияет только на span-метрику. Это также контроль корректности абляции.\n\n![F1 по типам](reports/hw2/figures/class_f1.png)\n\n![Матрицы ошибок](reports/hw2/figures/confusion.png)\n\nМатрицы относятся к классификации кандидатов; они не измеряют ошибки границ.\n\n"
    slices = load(root / "length_slices.json")
    text += "Срезы по длине эссе (одинаковые группы для всех конфигураций):\n\n"
    text += table(["Срез", *names], [[s, *[f'{slices[n][s]["macro_f1"]:.4f}' for n in names]] for s in slices["baseline"]])
    text += "\n\nСостав классов в срезах различается, поэтому длина не считается доказанной причиной различий. Локальные примеры исправленных, новых и сохраняющихся ошибок: `reports/hw2/private/error_examples.md`. Их тексты не включены в архив.\n\n"
    losses = []
    for n in ["context_only", "decoder_only", "context_decoder"]:
        for c in CLASSES:
            before, after = results["baseline"]["per_class"][c]["f1"], results[n]["per_class"][c]["f1"]
            if after < before:
                losses.append([n, c, f"{before:.4f}", f"{after:.4f}", f"{after - before:+.4f}"])
    if losses:
        text += "**Что не сработало по отдельным классам:**\n\n"
        text += table(["Вариант", "Класс", "Baseline F1", "Новый F1", "Δ"], losses)
        text += "\n\nРучное правило коротких ролей нельзя считать универсальным: сохранение каждой кандидатной границы может дробить один gold-аргумент. Это согласуется с наблюдаемым ухудшением, но не доказывает его причину. Таблица описательна, поклассовые CI не строились.\n\n"
    text += "## 5. Честность сравнения\n\n"
    text += f'- Три одинаковых essay/group-фолда, seed=42. Сверка с локальным split ДЗ1: {checks["same_split_as_hw1"]}. Полный нормализованный дубликат остаётся в одной группе.\n'
    text += '- Страты — наличие Counterclaim/Rebuttal × длина эссе. Две составные страты содержат по одному эссе, поэтому sklearn предупреждает о невозможности точного сохранения всех пропорций. Split оставлен прежним ради сравнения; все семь целевых типов присутствуют во всех фолдах. Контрольные агрегаты: [числа классов по фолдам](reports/hw2/fold_class_counts.csv) и [числа страт](reports/hw2/stratum_counts.csv).\n'
    text += f'- Baseline переобучен, а не взят из таблицы: разница pooled OOF macro F1 с ДЗ1 = {checks["hw1_baseline_oof_difference"]}.\n'
    text += "- Одинаковые исходные файлы, word-индексы `.split()`, эвристические кандидаты и train-targets. Gold-границы не подаются на inference.\n- TF-IDF/IDF, DictVectorizer, class weights и SGD обучаются только на train фолда. На val только transform/predict. Контекст на val строится внутри целого val-эссе и не обращается к train-эссе.\n- Preprocessing фиксирован; экспериментальные контекстные блоки и decoder явно обозначены как единственные изменения. Иначе проверка нового компонента была бы невозможна.\n- Одна и та же основная span macro F1, та же реализация matching, одни гиперпараметры SGD, без HPO. Совпадение допустимо только при обоих overlap >=0.5; после однозначного matching считаются TP/FP/FN и F1 каждого из семи типов, затем среднее.\n- Выбор только по среднему validation macro F1. Test/leaderboard не участвуют. Полное обучение после выбора служит только inference.\n- 1000 парных cluster-bootstrap повторов по группам нормализованных эссе, seed=42; интервалы pooled OOF F1 и разностей — перцентили 2,5/97,5. Mean/std относятся к фолдам, CI — к ресемплингу OOF-групп.\n\n"
    text += "Ограничения: оценки разных фолдов используют пересекающиеся train-части; bootstrap не учитывает переобучение и selection bias. Гипотезы опираются на уже изученную CV из ДЗ1. Эти интервалы условны относительно протокола, а не являются независимой итоговой оценкой. Author/prompt-ID отсутствуют, near-duplicates полностью не контролируются. Положительный CI не доказывает перенос на новые темы.\n\n"
    text += "## 6. Выводы\n\n"
    for n, key in [("context_only", "context_only_minus_baseline"), ("decoder_only", "decoder_only_minus_baseline")]:
        d = comparisons[key]
        verdict = {"positive": "гипотеза о росте macro F1 поддержана", "negative": "гипотеза о росте macro F1 не поддержана, получено ухудшение", "inconclusive": "устойчивый прирост не установлен"}[d["interpretation"]]
        text += f'- **{n}:** {verdict}; Δ={d["delta_oof"]:+.4f}, CI {interval(d["paired_bootstrap_ci95"])}.\n'
    text += f'- **Выбор:** `{selected}` по среднему validation macro F1, без перенастройки после просмотра результатов.\n'
    text += "- Контекст может помогать за счёт соседних лексических маркеров, но причинный механизм этими экспериментами не установлен. Избирательное объединение меняет компромисс между пропущенными и лишними spans: короткие роли могут выиграть в recall, длинные — потерять из-за дробления.\n- Эффекты с CI, включающим ноль, считаются неразличимыми на этом split. Устойчивость к новым seed и повторному обучению отдельно не проверена. Проверенные изменения могут давать общий прирост, сохраняя неудачные поклассовые эффекты из раздела 4.\n- Следующий тест: отдельная обучаемая модель границы между кандидатами вместо ручного списка объединяемых типов. Сначала фиксируется протокол независимой оценки; ещё один подбор списка классов по этой CV не планируется.\n\n"
    text += "## 7. Запуск и артефакты\n\nPython 3.9–3.12, CPU, зависимости из `requirements.txt`. Папка `feedback-prize-2021/` располагается в корне проекта.\n\n```bash\npython3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements.txt\nbash scripts/run_hw2.sh\n```\n\nОтдельные этапы:\n\n```bash\npython3 -m unittest discover -s tests -v\npython3 -m src.hw2_train --config configs/hw2.json\npython3 -m src.hw2_report\npython3 -m src.hw2_predict --output submission_hw2.csv\n```\n\nДля независимого пересчёта span-метрики можно использовать `src.evaluate` с любым `reports/hw2/predictions/*_oof.csv`. OOF-predictions CSV, split, checkpoint и примеры с текстами сохраняются локально. Сдаваемые агрегированные метрики и конфиги лежат в `reports/hw2/` и `configs/hw2.json`. Ни один файл результатов ДЗ1 запуск не перезаписывает.\n\n"
    text += "## 8. Источники и реализация\n\n[Условия ДЗ2](https://github.com/PikaChuChuMF/RnD-ML-Course-2026-2/blob/main/HW_2/HomeWork_2.md), [определение метрики соревнования](https://www.kaggle.com/competitions/feedback-prize-2021/overview/evaluation), [SGDClassifier 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.SGDClassifier.html), [TF-IDF 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). Используются код baseline и метрики ДЗ1 и стандартные реализации sklearn. Контекст, selective decoder, факторный план, трекинг и отчёт реализованы для проекта с помощью Codex. Чужие notebooks, внешние данные и готовые веса не использованы.\n"
    Path("HW2.md").write_text(text, encoding="utf-8")

    # Keep student texts in the ignored local directory only.
    frame = pd.read_pickle(root / "private" / "candidates_oof.pkl")
    fixed = (frame.baseline_prediction != frame.target) & (frame.context_prediction == frame.target)
    harmed = (frame.baseline_prediction == frame.target) & (frame.context_prediction != frame.target)
    still = (frame.baseline_prediction != frame.target) & (frame.context_prediction != frame.target)
    examples = "# OOF-примеры влияния контекста\n\n"
    for heading, mask in [("Исправленная ошибка", fixed), ("Новая ошибка", harmed), ("Ошибка сохранилась", still)]:
        subset = frame.loc[mask & ((frame.end - frame.start).between(8, 50))]
        if len(subset):
            r = subset.sort_values(["id", "start"]).iloc[0]
            examples += f'## {heading}\n\nЭссе `{r.id}`, fold={r.fold}, слова [{r.start}, {r.end}). Gold-класс кандидата: {r.target}; baseline: {r.baseline_prediction}; context: {r.context_prediction}.\n\n> {r.text.replace(chr(10), " ")}\n\n'
    examples += "Классы здесь — доминирующая gold-метка кандидата. Примеры иллюстрируют классификацию; итоговые границы и качество извлечения оцениваются span F1.\n"
    (root / "private" / "error_examples.md").write_text(examples, encoding="utf-8")
    print("Created HW2.md, experiments.csv, four figures and private error examples")


if __name__ == "__main__":
    main()
