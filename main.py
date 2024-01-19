# Celebrity Search - OpenAI Integration

# Import necessary libraries and modules
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Celebrity Search')
input_text = st.text_input('Enter The Name Of The Celebrity')

# Define prompt templates for language model queries

# Prompt for celebrity biography
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# Memory objects to store conversation history
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# Create an instance of the OpenAI language model
llm = OpenAI(temperature=0.8)

# Create language model chains for sequential queries

# Chain 1: Retrieve celebrity biography
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)

# Chain 2: Retrieve celebrity birthdate
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)

# Chain 3: Retrieve major events around the celebrity's birthdate
third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="mention 5 major events happened around {dob} in the world"
)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=descr_memory)

# Create a sequential chain to execute the queries in order
parent_chain = SequentialChain(chains=[chain, chain2, chain3], input_variables=['name'],
                               output_variables=['person', 'dob', 'description'], verbose=True)

# Check if user input is provided
if input_text:
    # Execute the chain and display results
    st.write(parent_chain({'name': input_text}))

    # Display conversation history in expandable sections
    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)
