# InfraMIND Proteus - A predictive model of dengue cases

InfraMIND Proteus is the first model of the InfraMIND (Infrastructure for Modeling Infectious Diseases) family, specifically developed for the 3rd edition of the [Infodengue-Mosqlimate Dengue Challenge](https://sprint.mosqlimate.org/) (IMDC).


# Quick start

Requirements:
- python
- [Astral UV](https://docs.astral.sh/uv/)
- git

Create a fork and clone it locally:
```bash
git clone https://github.com/<your_username>/3rd_imdc_ifgw_inframind-proteus.git
cd 3rd_imdc_ifgw_inframind-proteus
```

Setup the development environment (assumes you have uv).
Follow the instructions as prompted.
```
python3 setup_env.py
```

Activate the environment:
```
source .venv/bin/activate
```

Re-activate the environment every new command line session. Alternatively, you can run without activating the environment by calling `uv run [commands]`.
For example, running
```
uv run jupyter notebook
```

will start a jupyter server already configured for the Proteus model.





# Required information

These topics are required by the organization rules and will be filled over time during the sprint.

1. Team and Contributors

    Name of your team.
    Names of all contributors and their affiliations >(universities/institutions, if applicable).

2. Repository Structure

A brief description of the contents and purpose of >each folder and file in the repository.

3. Libraries and Dependencies

A list of all libraries and packages used to process the data and train your model.

4. Data and Variables

    Which datasets and variables were used?
    How was the data pre-processed?
    How were the variables selected? Please point to the relevant part of the code.

5. Model Training

    Description of how the model was trained. If applicable, describe any hyperparameter optimization techniques used.

    Please specify where the code for training and generating forecasts is located, and provide instructions on how to run it.

5. Data Usage Restriction

Describe how you handled the requirement of using only data up to EW 25 of the current year to generate predictions from EW 41 of the same year to EW 40 of the next year.

6. Predictive Uncertainty How are your prediction intervals computed?

7. References

If your model is based on a published or preprint (e.g., arXiv) paper, include the citation, DOI, and link.
