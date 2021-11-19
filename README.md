
# Recrutation - KN Robocik (Wiktor Sadowy)

Recruitment task (task number 2) for KN Robocik

The descryption of the task (in Polish language) is in the file "Opis projektu.pdf"

The analysis of the results is in the file "Results analysis.pdf"


## Installation

Copy the project using git

```bash
  git clone https://github.com/WiktorSa/KN-Robocik-Recrutation
  cd KN-Robocik-Recrutation
```

## How to run the project

To get preprocessed data run

```bash
  python get_preprocessed_data.py
```

To train the regression model run

```bash
  python train_model.py -model regression
```

To train the classification model run

```bash
  python train_model.py -model classification
```

To see the results run

```bash
  python show_results.py
```

To see what parameters you can modify run:

```bash
  python get_preprocessed_data.py --help
  python train_model.py --help
  python show_results.py --help
```

## Authors

- [@WiktorSadowy](https://github.com/WiktorSa)

