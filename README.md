## Uruchomienie
Do wykonania całego działania należy nadać potrzebne uprawnienia oraz wykonać jeden skrypt:
```bash
chmod +x setup_environment.sh
chmod +x train_and_evaluate.sh
chmod +x evalF1_no_penalty.pl
chmod +x evalF1_penalty.pl
chmod +x wellformed.pl
./train_and_evaluate.sh
```

Skrypt najpierw przygotuje środowisko poprzez stworzenie wirtualnego środowiska
Python oraz pobranie wszystkich potrzebnych bibliotek. Następnie przygotuje dane
oraz przeprowadzi cały proces trenowania modelu oraz ewaluacji.

W projekcie załączono dwa skrypty w języku Python. Skrypt `prepare_data.py` przygotowuje
dane w formacie XML do przeczytania podczas trenowania. Drugi skrypt `main.py` uruchamia
całą funkcjonalność projektu według podanych argumentów. Dostępne argumenty można
wyświetlić dodając flagę `-h`, przykładowo:
```commandline
> python main.py -h
usage: main.py [-h] [--train-paths TRAIN_PATHS] [--test-path TEST_PATH]
               [--model MODEL] [--classifier-model-name CLASSIFIER_MODEL_NAME]
               [--batch-size BATCH_SIZE] [--epochs EPOCHS]
               [--learning-rate LEARNING_RATE]

CLI wrapper for NLP classifier using BERT

optional arguments:
  -h, --help            show this help message and exit
  --train-paths TRAIN_PATHS
                        comma separated paths for the train datasets
  --test-path TEST_PATH
                        path for the test dataset
  --model MODEL         which model will be trained and evaluated
  --classifier-model-name CLASSIFIER_MODEL_NAME
                        name of the sentence transformer model
  --batch-size BATCH_SIZE
                        size of mini batch used during training
  --epochs EPOCHS       number of epochs
  --learning-rate LEARNING_RATE
                        learning rate used during training
```
