# EducationFactory - Course on MLOps 
## Setting up the environment for local usage

1. Install miniconda (see https://docs.conda.io/en/latest/miniconda.html)

2. Navigate to the repo directory and open a command prompt in this directory 

3. Recreate the virtual environment using conda:
   
   `conda env create -p ./.venv -f environment.yml`
   
   Note that using the code above assumes that the name of your virtual environment will be `venv`
    
    To recreate the river_environment:

    `conda env create -p ./.river_venv -f environment_river.yml`

4. Activating the environment can be done by running:

	`conda activate ./.venv`
   
5. Add the virtual environment as interpreter in PyCharm

6. Packages can be installed in the environment through the command line:

	`conda install -p ./.venv <my-desired-package>`
	
7. If the environment file is updated, the local virtual environment can be updated using:

   `conda env update -p ./.venv -f environment.yml  --prune`

### Add config to easily access workspace

1. Go to the [AzureML Workspace for this course](https://ml.azure.com/?wsid=/subscriptions/cc63ab5a-0493-449c-9a0d-9d46ed294079/resourceGroups/rg-mlops-demo/providers/Microsoft.MachineLearningServices/workspaces/ml-mlops-demo&tid=78ff5534-7a04-4798-979b-a46c61bc0fc0) 

2. On the top right of the page press the dropdown menu with the name of the workspace

3. Press "Download config file"

4. Save this json file as `config.json` in the project folder you are working on 

5. Now you can access the workspace in your python code using:
   ```
   from azureml.core import Workspace
   ws = Workspace.from_config()
   ```
  
6. Make sure that when running a notebook, the `config.json` file is in your working directory! Otherwise the lines above will not work. 

**Now you can connect to the AzureML workspace to submit your experiments**