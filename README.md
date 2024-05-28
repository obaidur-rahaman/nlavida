# Natural Language-Assisted Visualization and Interactive Data Analysis (NLAVIDA)

<p align="center">
  <img src="/assets/logo.jpg" alt="Logo" width="20%">
</p>

## Introduction Video

[![Watch the video](https://img.youtube.com/vi/leJRP_mJsSQ/0.jpg)](https://www.youtube.com/watch?v=leJRP_mJsSQ&t=4s)

## Description

This open-source tool provides a robust solution for data analysis and visualization, designed specifically for use within enterprise environments where confidentiality is paramount. It utilizes agents and tools to process structured data exclusively. Users can securely upload and manipulate files. Unlike black-box solutions from other service providers, this tool offers full transparency and allows for extensive customization to meet specific internal needs. Leveraging the power of OpenAI's GPT-4o, it enhances data processing capabilities, making it an indispensable resource for researchers, data scientists, and business analysts working with confidential information.

## Key Features

### Confidentiality
- **Secure File Upload**: Safely upload and process confidential files.
- **Integration with OpenAI GPT-4o**: Leverage GPT-4o for secure data analysis and manipulation.
- **Enterprise-Ready**: Compliant with internal data security protocols.

### Customizability
- **Flexible Architecture**: Extensively customizable to fit specific internal processes.
- **Advanced Analytics & Code Execution**: Utilize sophisticated algorithms and GPT-4o for deep data insights, with direct code execution for robust scripting.
- **In-House Solutions**: Customizable to build comprehensive in-house solutions to meet all internal needs.

### Cost
- **Scalability**: Scales with organizational needs, ensuring cost-effective growth.
- **Cost Efficiency**: Eliminates the need for subscriptions to external service providers. Future versions will use free and open-source LLMs, reducing costs further.


## Installation

#### Clone repository 

To get started with this project, clone the repository to your local machine and navigate to the source code directory:

you can either use HTTP:
```bash
git clone https://github.com/obaidur-rahaman/nlavida.git
cd nlavida/src
```

or SSH:
```bash
git clone git@github.com:obaidur-rahaman/nlavida.git
cd nlavida/src
```

#### Generate API key and setup environment 

To use the OpenAI API, you need an API key, which is used to authenticate requests to the API. Here’s how to obtain your API key:

Visit the website https://platform.openai.com/api-keys (create an account if you do not have one) and generate a new secret key. For best practices on handling your API key, please refer to: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key.

Next, copy the example environment file and enter your API key:

```bash
cp .envExample .env
```

Use your favorite editor to open the .env file and replace your_OpenAI_api_key with the OpenAI API key you just created.

#### Set up Conda

To set up the necessary Conda environment to run this project with Miniforge, follow these steps:

If you do not have Miniforge installed, download and install it from the Miniforge GitHub Releases page, following the installation instructions for your operating system.

Create and activate the environment:

```bash
conda env create -f ../environment.yaml
conda activate nlavida
```

#### Start the application

You can now start your application:

```bash
python app.py
```

Once it starts, you can access the application by typing the following address in your browser:

http://127.0.0.1:8002

## Support

If you encounter any problems with the installation or usage, please feel free to contact me at obaidur.rahaman345@gmail.com.

## Roadmap

This project is under continuous development. I plan to work on the following extensions in the near future:

- Incorporation of other LLMs like llama3 and Azure Open AI.
- Ability to perform SQL queries to retrieve data from internal databases.
- Integration of unstructured data (using RAG) with structured data.

## Contributing

Your help is needed to make this tool more useful for the open source community. Contributions to this source code are very welcome.

If you have ideas for extending this tool or want to collaborate, please reach out to me via email at obaidur.rahaman345@gmail.com. I am confident it will be a great learning experience for all participants and a positive contribution to the community.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


