# Financial Markets Prediction AI Program

## Overview

This Python program is designed to predict the EUR/USD exchange rate using historical tick data from Dukascopy and Google Cloud's Vertex AI. It aims to forecast the currency pair's performance for the next 12 months and outputs the predictions to a .csv file for further analysis.

## Prerequisites

To run this program, you will need:

1. A Google Cloud Platform (GCP) account with billing enabled and the necessary permissions to access Vertex AI services.
2. Historical tick data for the EUR/USD currency pair.
3. Python 3.x installed on your system.
4. Google Cloud SDK installed and authenticated with your GCP project.
5. The following Python libraries installed:

   ```
   google-cloud-aiplatform
   pandas
   scikit-learn
   ```

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/financial-market-predictor.git
   ```

2. Navigate to the project directory:

   ```
   cd financial-market-predictor
   ```

3. Install the required Python libraries:

   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Set up your GCP credentials as per the Google Cloud documentation.
2. Prepare your Dukascopy tick data in a tabular format with appropriate columns.
3. Update the `config.json` file with your GCP project ID, location, dataset name, model name, and data file path.

## Usage

Run the program with the following command:

```
python financial_market_predictor.py
```

## Program Structure

- `config.json`: Contains configuration settings for the GCP project, dataset, model, and file paths.
- `requirements.txt`: Lists all the necessary Python libraries.
- `financial_market_predictor.py`: The main script that preprocesses data, trains the model, and makes predictions.
- `LICENSE`: The MIT License for the project.
- `README.md`: This file, which provides an overview and instructions for the project.

## Customization

You can customize the model by adjusting the hyperparameters in the `config.json` file or by implementing additional feature engineering techniques in the `financial_market_predictor.py` script.

## Support

For any issues or questions, please create an issue on the GitHub repository or contact [your@email.com](mailto:your@email.com).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
