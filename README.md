# CORTEX-Health

CORTEX-Health is an initiative towards automating disease diagnosis process using machine learning models.

## Installation Instruction

- Clone the repository: `git clone https://github.com/abrar-nazib/cortex-health`
- Navigate to the repository and create a virtual environment with venv `python -m venv venv`
- Activate the environment
  - For windows: `.\venv\Scripts\activate`
  - For linux: `source venv/bin/activate`
- Install the required packages: `pip install -r requirements.txt`
- Run the server `uvicorn app.main:app --reload`
  - It will return the server ip and port in `IP:PORT` format
  - Usually the IP is 127.0.0.1 and the port is 8000

## API testing

For the API documentation, after running the server, go to the `http://IP:PORT/docs` and create requests accordingly
