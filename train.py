import mlflow
import mlflow.tensorflow
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Pneumonia Detection CNN")

mlflow.tensorflow.autolog()

best_acc_file = "best_accuracy.txt"
current_best = 0

if os.path.exists(best_acc_file):
    with open(best_acc_file, "r") as f:
        current_best = float(f.read())

with mlflow.start_run():

    # your training code...
    # after evaluation:
    new_acc = acc  # your test accuracy

    if new_acc > current_best:
        with open(best_acc_file, "w") as f:
            f.write(str(new_acc))

        with open("model_improved.txt", "w") as f:
            f.write("YES")
    else:
        with open("model_improved.txt", "w") as f:
            f.write("NO")