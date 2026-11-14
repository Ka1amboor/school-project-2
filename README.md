# Лабораторная работа 2 — Анализ сходства и оценка качества

**Траектория: Python Core. Уровень: основной. Python 3.12+.**

Продолжение ЛР1: сохранены приём, валидация и неизменяемые версии сдач.
Добавлены ленивый конвейер JSONL → validate → features, token 3-граммы + Jaccard,
сравнение нормализованных AST, проверяемые основания и оценка на раздельных семействах.
Runtime использует только стандартную библиотеку Python. 

## Запуск

Из корня распакованного проекта:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e . -r requirements-dev.txt
python scripts/reproduce.py
```

Полное воспроизведение последовательно запускает генератор фиксированного
корпуса, тесты, lint, mypy, оценку качества, выгрузку пар, демонстрацию нового шаблона
и измерения до/после оптимизации: 100/1000/5000 записей, по три повтора.
Генератор перезаписывает только поставляемый синтетический корпус `data/lab2/`.

Быстрая проверка: `python scripts/reproduce.py --skip-benchmark`.

В готовой поставке также есть wheel. Для запуска без загрузки зависимостей:

```bash
python -m pip install --no-index wheels/school_lab-0.2.0-py3-none-any.whl
python -m school_lab --help
```

Для тестов нужны dev-зависимости; основной анализ работает без них.

## Основные команды

```bash
# ЛР1: прежний импорт
school-lab import data/submissions.jsonl

# все допустимые пары, два метода, JSONL-отчёты с evidence
school-lab analyze data/lab2/submissions.jsonl \
  --assignments data/lab2/assignments.json --config data/lab2/config.json \
  --output results/lab2/analysis

# только top-5: исключены сама работа, другое задание и версии того же студента
school-lab analyze data/lab2/submissions.jsonl \
  --assignments data/lab2/assignments.json --method ast \
  --submission-id family-08-0 --top-k 5 --threshold 1.0 \
  --output results/lab2/top5

# порог выбирается только по tune; evaluation не участвует в подборе
school-lab evaluate data/lab2/submissions.jsonl \
  --assignments data/lab2/assignments.json --config data/lab2/config.json \
  --provenance data/lab2/provenance.jsonl --pairs data/lab2/pairs.jsonl \
  --output results/lab2

python scripts/benchmark_lab2.py
python scripts/demo_template.py
python -m pytest -q
python -m ruff check .
python -m mypy src
```




