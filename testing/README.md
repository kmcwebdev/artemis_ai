# Kedro -  Pipeline in Data Processing and Model Training

Kedro is an open-source framework to manage a data science project.
It provides a pipeline visualization to keep track of machine-learning experiments, and makes it easier to collaborate with business stakeholders.

## Overview

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`


## Run Locally

Clone the project and go to the `kedro-test` branch

```bash
  git clone https://github.com/kmcwebdev/artemis_ai.git
```

Go to the project directory

```bash
  cd testing
```
### Set up environment
##### Creating a Virtual Environment with venv
1/ To create an environment:
```bash
  python3.10 -m venv myenv
```
2/ Activate environment
##### On macOS and Linux:
```bash
  source myenv/bin/activate
```
##### On Windows:
```bash
  myenv\Scripts\activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```

To view the pipeline
```bash
  kedro viz run
```
To run the nodes in each pipeline. Replace the node_name with some node name such as: ``, ``,``
```bash
  kedro run --nodes=node_name
```



## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
