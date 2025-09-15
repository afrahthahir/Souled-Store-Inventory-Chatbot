# SQL Database Chatbot
This project implements an intelligent chatbot that can answer questions by querying a SQL database using Natural Language Processing (NLP). Built with LangChain, it connects to your database, understands user queries, converts them into SQL, executes the queries, and provides natural language answers based on the retrieved results.

## Features
*Natural Language to SQL*: Translates user questions into executable SQL queries.

*Database Interaction*: Directly interacts with a configured SQL database.

*Intelligent Answering*: Generates human-like answers based on the SQL query results.

*LangChain Integration*: Leverages LangChain's SQLDatabase chain for robust functionality.

## How It Works
The chatbot uses a LangChain SQLDatabase chain, which orchestrates the following process:

*User Query*: The user asks a question in natural language (e.g., "What are the names of all t-shirts from the 'Nike' brand?").

*Schema Understanding*: The LangChain SQLDatabase chain inspects the connected SQL database schema to understand table names, column names, and relationships.

*SQL Generation*: A Large Language Model (LLM) (e.g., Google Gemini Flash) analyzes the user's question and the database schema to generate an appropriate SQL query.

*SQL Execution*: The generated SQL query is executed against the database.

*Result Interpretation*: The results from the SQL query are then fed back into the LLM.

*Natural Language Answer*: The LLM interprets the SQL results and formulates a human-readable answer for the user.

## Setup
Follow these steps to get your chatbot running on your local machine.

## Prerequisites
Python 3.8+

A SQL database (e.g., SQLite, PostgreSQL, MySQL) with your data loaded. This README assumes you have a database file or connection details.

A Google API Key (for the LLM).