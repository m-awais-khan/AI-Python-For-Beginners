from openai import AzureOpenAI, DefaultHttpxClient

import pandas as pd
from IPython.display import display, HTML

client = AzureOpenAI(
    api_key="abcdefg",
    api_version="2024-02-01",
    azure_endpoint = "https://cour-external-playground.openai.azure.com/",
    http_client=DefaultHttpxClient(verify=False)
    )


# +
#

# ### If you want to use your own OpenAI key, uncomment these lines below

# from openai import OpenAI

# ### Add your key as a string
# openai_api_key = "Add your key in here"

# # Set up the OpenAI client
# client = OpenAI(api_key=openai_api_key)

#
# -

def print_llm_response(prompt):
    """This function takes as input a prompt, which must be a string enclosed in quotation marks,
    and passes it to OpenAI's GPT 4o-mini model. The function then prints the response of the model.
    """
    try:
        if not isinstance(prompt, str):
            raise ValueError("Input must be a string enclosed in quotes.")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful but terse AI assistant",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        response = completion.choices[0].message.content
        print(response)
    except TypeError as e:
        print("Error:", str(e))


def get_llm_response(prompt):
    """This function takes as input a prompt, which must be a string enclosed in quotation marks,
    and passes it to OpenAI's GPT 4o-mini model. The function then saves the response of the model as
    a string.
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful but terse AI assistant",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    response = completion.choices[0].message.content
    return response


def display_table(data):
    df = pd.DataFrame(data)

    # Display the DataFrame as an HTML table
    display(HTML(df.to_html(index=False)))

