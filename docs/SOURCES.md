# Источники и заимствования

1. [Условия ДЗ1](https://github.com/PikaChuChuMF/RnD-ML-Course-2026-2/blob/main/HW_1/HomeWork_1.md)
   — требования к постановке, baseline, валидации и оформлению.
2. [Описание и оценка Feedback Prize](https://www.kaggle.com/competitions/feedback-prize-2021/overview/evaluation)
   — постановка segmentation + classification, семь типов, формат
   `id,class,predictionstring`, индексы `text.split()` и macro F1
   с двунаправленным overlap >=0.5.
3. [Данные соревнования](https://www.kaggle.com/competitions/feedback-prize-2021/data)
   — тексты и разметка в `feedback-prize-2021/`. Размеры и статистики
   рассчитаны по локальным файлам.
4. [Правила соревнования](https://www.kaggle.com/competitions/feedback-prize-2021/rules)
   — условия использования данных.
5. Документация scikit-learn:
   [TF-IDF](https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
   [SGDClassifier](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.SGDClassifier.html),
   [StratifiedGroupKFold](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
   [DummyClassifier](https://scikit-learn.org/1.6/modules/generated/sklearn.dummy.DummyClassifier.html).
   Использованы стандартные библиотечные реализации этих методов.

Сегментация, обработка конфликтной разметки, decoder, split, абляции,
span matching, bootstrap, проверки и отчёт реализованы для проекта с помощью
Codex. Matching написан по определению метрики соревнования. Чужие Kaggle
notebooks, решения участников, готовые веса и внешние данные не использованы.
