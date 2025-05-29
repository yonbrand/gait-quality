import hydra
import time
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import TestDataset, setup_model, test_model, evaluate_metrics, compute_icc, plot_correlation_cohort
from datetime import datetime

now = datetime.now()


@hydra.main(config_path="conf", config_name="config_test", version_base='1.1')
def main(cfg):
    # weights_path = cfg.model.weights_path
    output_path = cfg.data.log_path
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = setup_model(
        net= cfg.model.net,
        head= cfg.model.head if cfg.model.net == 'ElderNet' else None,
        eldernet_linear_output=cfg.model.linear_output_size if cfg.model.net == 'ElderNet' else None,
        epoch_len=cfg.dataloader.epoch_len,
        is_regression=cfg.model.is_regression,
        max_mu=cfg.data.max_mu,
        num_layers_regressor=cfg.model.num_layers_regressor,
        batch_norm=cfg.model.batch_norm,
        trained_model_path=cfg.model.weights_path,
        device=device)

    model.to(device)

    # Load the data
    X = pickle.load(open(os.path.join(cfg.data.data_root, 'acc.p'), 'rb'))
    if X.shape[1] != 3 :
        X = np.transpose(X, (0, 2, 1))
    subjects = pickle.load(open(os.path.join(cfg.data.data_root, 'subjects.p'), 'rb'))

    # Load labels if exist
    if cfg.data.labels is not None:
        labels = pickle.load(open(os.path.join(cfg.data.data_root, cfg.data.labels), 'rb'))
        test_dataset = TestDataset(X, labels, groups=subjects)
    else:
        test_dataset = TestDataset(X, groups=subjects)


    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=1,
    )

    # Test the model
    start = time.time()
    results = test_model(model, test_loader, device)
    end = time.time()
    print(end-start)

    # Calculate metrics
    if cfg.model.plot:
        true_labels, predictions = results[0], results[1]
        mae, rmse,  mape, r2 = evaluate_metrics(true_labels, predictions)
        icc = compute_icc(true_labels, predictions)

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"ICC: {icc:.4f}")

        # Plot and save correlation
        plot_path = os.path.join(output_path, f"correlation_plot_{now.strftime('%Y%m%d_%H%M%S')}.png")
        plot_correlation_cohort(true_labels,
                                predictions,
                                measure=cfg.data.measure,
                                unit=cfg.data.unit,
                                title=cfg.data.title,
                                save_path=plot_path)

        print(f"Correlation plot saved to: {plot_path}")

        with open(os.path.join(output_path, 'labels.p'), 'wb') as f:
            pickle.dump(true_labels, f)

    else:
        predictions = results

    with open(os.path.join(output_path, 'predictions.p'), 'wb') as f:
        pickle.dump(predictions, f)


if __name__ == '__main__':
    main()