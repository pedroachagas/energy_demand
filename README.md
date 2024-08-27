# Electricity Demand Forecasting App

This project is an end-to-end electricity demand forecasting application that uses machine learning to predict future electricity demand based on historical data.

## Project Structure

```plaintext
electricity-demand-forecasting/
├── .github/workflows/  # GitHub Actions workflows
├── ci/                 # CI/CD specific files
├── src/                # Main application code
├── notebooks/          # Jupyter notebooks for analysis
├── tests/              # Unit and integration tests
├── .streamlit/         # Streamlit configuration
├── .env.example        # Template for environment variables
├── Makefile            # Project management commands
├── README.md           # This file
└── requirements.txt    # Project dependencies
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/electricity-demand-forecasting.git
   cd electricity-demand-forecasting
   ```

2. Set up the project:

   ```bash
   make setup
   ```

3. Set up environment variables:
   Copy `.env.example` to `.env` and fill in your specific values.

## Usage

The project uses a Makefile for common tasks. Here are the available commands:

- Set up the project: `make setup`
- Run ETL pipeline: `make run-etl`
- Run scoring pipeline: `make run-scoring`
- Run Streamlit dashboard: `make run-dashboard`
- Run tests: `make test`
- Remove virtual environment: `make clear`
- Clean up generated files and virtual environment: `make clean`
- Reset the project (clean and setup): `make reset`
- Install dependencies: `make install`
- Update requirements.txt: `make freeze`

## Development

- Add tests to the `tests/` directory
- Use `notebooks/` for exploratory data analysis
- Update `requirements.txt` when adding new dependencies (you can use `make freeze`)

## CI/CD

The project uses GitHub Actions for CI/CD:

- `modeling.yml`: Builds a Docker image and runs the modeling pipeline
- `pipeline.yml`: Runs the ETL and scoring pipelines daily

The Dockerfile for the modeling pipeline is located in `ci/Dockerfile`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
