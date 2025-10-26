import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import(
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, model, test_loader, classes, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.classes = classes
        self.device = device

        self.all_labels = []
        self.all_preds = []
        self.all_probs = []

    def evluate(self):
        self.model.eval()
        print("Evaluating model on test set")
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)

                _, preds = torch.max(outputs,1)

                self.all_labels.extend(labels.cpu().nupy())
                self.all_preds.extend(preds.cpu().numpy())
                self.all_probs.extend(probs.cpu().numpy())

        self.all_labels = np.arrary(self.all_labels)
        self.all_preds = np.array(self.all_preds)
        self.all_probs = np.array(self.all_probs)

        print(f"Evaluated {len(self.all_labels)} samples")

    def plot_confusion_matrix(self, normalize=False):
        cm = confusion_matrix(self.all_labels, self.all_preds)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            ft = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self):
        n_classes = len(self.classes)
        plt.figure(figsize=(10,8))

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(self.all_labels, self.all_probs[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.3f})')

        else:
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(
                    (self.all_labels == i).astype(int),
                    self.all_probs[:, i]
                )
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2,
                         label=f'{self.classes[i]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_precission_recall_cure(self):
        n_classes = len(self.classes)
        plt.figure(figsize=(10,8))

        if n_classes == 2:
            precision, recall, _ = precision_recall_curve(
                self.all_labels, self.all_probs[:, 1]
            )
            avg_precision = average_precision_score(
                self.all_labels, self.all_probs[:, 1]
            )

            plt.plot(recall, precision, color='darkorange', lw=2,
                     label=f'PR curve (AP = {avg_precision:.3f})')
        else:
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(
                    (self.all_labels == i).astype(int),
                    self.all_probs[:, i]
                )
                avg_precision = average_precision_score(
                    (self.all_labels == i).astype(int),
                    self.all_probs[:, i]
                )
                plt.plot(recall, precision, lw=2,
                         label=f'{self.classes[i]} (AP = {avg_precision:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_classification_report(self):
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.classes,
            digits=4
        ))

    def plot_confidence_distribution(self):
        confidences = np.max(self.all_probs, axis=1)

        correct_mask = (self.all_labels == self.all_preds)
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.boxplot([correct_conf, incorrect_conf],
                    labels=['Correct', 'Incorrect'])
        plt.ylabel('Confidence Score')
        plt.title('Confidence Score by Prediction Correctness')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nMean confidence (correct): {correct_conf.mean():.4f}")
        print(f"Mean confidence (incorrect): {incorrect_conf.mean():.4f}")

    def analyze_errors(self, num_examples=5):

        error_indices = np.where(self.all_labels != self.all_preds)[0]

        if len(error_indices) == 0:
            print("No errors found!")
            return

        print(f"\nFound {len(error_indices)} misclassified samples")
        print(f"Error rate: {len(error_indices) / len(self.all_labels) * 100:.2f}%")

        sample_indices = np.random.choice(
            error_indices,
            min(num_examples, len(error_indices)),
            replace=False
        )

        return sample_indices

    def generate_full_report(self):
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("=" * 70)

        if len(self.all_labels) == 0:
            self.evaluate()

        self.print_classification_report()

        print("\nGenerating confusion matrix...")
        self.plot_confusion_matrix(normalize=False)
        self.plot_confusion_matrix(normalize=True)

        print("Generating ROC curve...")
        self.plot_roc_curve()

        print("Generating Precision-Recall curve...")
        self.plot_precision_recall_curve()

        print("Analyzing confidence scores...")
        self.plot_confidence_distribution()

        print("\nPerforming error analysis...")
        error_indices = self.analyze_errors()

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)

        return error_indices