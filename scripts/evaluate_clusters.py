from pathlib import Path
import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    input_path = PROJECT_ROOT / "data/processed/scaled_features.csv"
    output_path = PROJECT_ROOT / "reports/metrics/cluster_evaluation.csv"

    df = pd.read_csv(input_path)

    X = df.drop(columns=["CustomerID"])

    results = []

    print(f"Customers: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print("\nEvaluating K-Means...")

    for k in range(2, 11):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
            max_iter=500,
        )

        labels = model.fit_predict(X)

        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)

        results.append(
            {
                "n_clusters": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "davies_bouldin_score": davies,
            }
        )

        print(
            f"K={k:2d} | "
            f"Inertia={model.inertia_:,.2f} | "
            f"Silhouette={silhouette:.4f} | "
            f"CH={calinski:,.2f} | "
            f"DB={davies:.4f}"
        )

    results_df = pd.DataFrame(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    best_silhouette = results_df.loc[
        results_df["silhouette_score"].idxmax()
    ]

    print("\nCLUSTER EVALUATION COMPLETE")
    print(f"Saved: {output_path}")

    print(
        f"\nBest silhouette K: "
        f"{int(best_silhouette['n_clusters'])}"
    )
    print(
        f"Best silhouette score: "
        f"{best_silhouette['silhouette_score']:.4f}"
    )

    print("\nEvaluation results:")
    print(results_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
