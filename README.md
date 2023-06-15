# Software

To streamline the attack prompt construction and evaluation process, we developed a software tool using the [Gradio library](https://gradio.app/). This software incorporates two primary features, namely, manual and automated attack prompt construction. Here, we describe the software's main interface, using the example detailed in the 'Algorithm Example' section of this document.

![Software interface of prep steps for generating an attack](./figure/Picture1.png)

The software has four components that users need to complete before generating an attack prompt:

1. **API Key Input Field:** Users input their API keys to access language model services (LLMs). Our software primarily utilizes ChatGPT (GPT-3.5-turbo) and GPT-4 for attack prompt construction and evaluation, but the code can be adjusted to accommodate other LLMs per user requirements.

2. **Question Type and Injection Element:** We differentiate between closed-ended and open-ended questions. Selecting a question type updates the default prompt templates, but users can also modify these templates in the associated input field. The inject item input field allows users to indicate the option or term that they wish to inject included in the LLM's responses.

3. **Position Field:** This allows users to place the prompt at the beginning, middle, or end of the question as per our study. Users can also designate a position for prompt injection in the survey question field. An option to omit prompt injection is available as well, allowing users to see how LLMs react to the question without any interference.

4. **Survey Question Input Field:** This is where users enter the question to be injected with attack prompts.

![Software interface of manual attack prompt construction](./figure/Picture2.png)

The manual attack prompt construction process includes the following steps:

- Users can either utilize the default attack prompt template or modify it to suit their requirements.
- Upon clicking the "Generate Attack Prompt" button, users can preview the newly created attack prompt and the corresponding survey questions imbued with the attack prompt.
- By pressing the "Evaluate Attack Prompt" button, users can assess the effectiveness of the injection across various evaluation LLMs and rounds, as per their needs.
- Additionally, users can retrieve detailed evaluation information by selecting the accordion button at the bottom, which provides specifics about the corresponding LLM's information, including the LLM data, response time, and response message, among other details.

![Software interface of automated attack prompt construction](./figure/Picture3.png)

The process for automated attack prompt construction consists of the following components:

1. **Problem Encoding Section:** This requires users to provide instructions and optional few-shot examples to guide LLMs in the task of attack prompt construction. We offer default prompts to aid users in understanding the expected format.

2. **Revision Section:** Here, users specify the number of revision rounds to be executed, and provide instructions for LLMs to revise their previous prompts. Users also select the evaluation metrics of interest (in this case, length and injection effectiveness) and decide on the use of COT prompting.

3. **Evaluation Section:** This aligns with the manual attack prompt construction process. Users select the LLM to evaluate prompt injection effectiveness and decide on the number of evaluation rounds.

Upon clicking the "Generate Attack Prompt" button, a summary table displaying the evaluation results for each automated constructed attack prompt and its corresponding prompt injection effectiveness will appear. Detailed construction history and evaluation information can be accessed by expanding the corresponding accordion buttons.
