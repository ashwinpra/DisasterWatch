from flask import Flask, request, jsonify, render_template
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, Tool
from langchain.utilities import SerpAPIWrapper
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from geopy.geocoders import Nominatim
import os 
from dotenv import load_dotenv  
import json
import sys 

# add a debug flag to the app which will be taken as command line argument
debug = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'debug':
        debug = True

app = Flask(__name__)

load_dotenv()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getdisasterDataFromIdea', methods=['POST'])
def get_disaster_data_from_idea():

    if debug: 
        # use a hardcoded response
        responses = [
            {
                "commentary": "A massive fire broke out at a chemical factory in Maharashtra's Pune district on Monday morning. The blaze erupted at the SVS Aqua Technologies plant in the Pirangut MIDC area around 7 am. At least 15 fire tenders were rushed to the spot to douse the flames. No casualties have been reported so far. The cause of the fire is not known yet.",
                "date": "2021-09-06",
                "source": "https://www.ndtv.com/india-news/massive-fire-breaks-out-at-chemical-factory-in-maharashtras-pune-2536164",
                "location": "Pune",
                "latitude": 18.5204,
                "longitude": 73.8567
            },
            {
                "commentary": "A massive earthquake of magnitude 7.2 struck Haiti on Saturday, killing at least 227 people and injuring more than 1,500 others. The quake, which was followed by a series of aftershocks, destroyed thousands of homes and buildings. The epicentre of the quake was about 125 km west of the capital Port-au-Prince. The quake was also felt in neighbouring Cuba and Jamaica.",
                "date": "2021-08-14",
                "source": "https://www.ndtv.com/world-news/magnitude-7-2-earthquake-strikes-haiti-227-killed-1-500-injured-250-000-affected-2518633",
                "location": "Haiti",
                "latitude": 18.9712,
                "longitude": 72.8015
            },
        ]


    else: 
        # Get the disaster idea from the request body
        data = request.get_json()
        disasterIdea = data.get('idea')

        # making the LangChain agent
        llm = OpenAI(temperature=0.9, openai_api_key=os.environ.get('OPEN_AI_KEY'))
        params = {
            "engine": "google",
            "gl": "us",
            "hl": "en",
            "domain": [ "ndtv.com", "bbc.in", "thehindu.com"], # not sure if this is working
        }
        search = SerpAPIWrapper(params=params) # search tool for the agent
        tool = Tool(
            name="search_tool",
            description="To search for relevant information about the disaster",
            func=search.run,
        )
        agent = initialize_agent([tool], llm, agent="zero-shot-react-description", verbose=True)

        # 1. Getting locations of disaster 

        # Structuring the prompt

        question = f"The user is interested in finding locations affected by disasters. Given the prompt {disasterIdea}, find the relevant affected areas of disaster (like specific landmarks, attractions, or sites), and return just the name of the locations in a list, separated by commas."
        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template("Answer the user's question as best as possible.\n{question}")  
            ],  
            input_variables=["question"],
        )

        input = prompt.format_prompt(question=question)

        # Run the agent
        locations = agent.run(input).split(',')

        print(locations)

        with open('locations.json', 'w') as f:
            json.dump(locations, f)


        # Getting the latitude and longitude of the locations
        geolocator = Nominatim(user_agent="your_app_name")
        locations_data = []
        for location in locations:
            try:
                geo_location = geolocator.geocode(location)
                locations_data.append({
                    'location': location,
                    'latitude': geo_location.latitude,
                    'longitude': geo_location.longitude,
                })
            except Exception as e:
                app.logger.info(f" Location not found or geocoding error: {location}")
                app.logger.info(f" Error: {e}")


        # 2. Getting commentary for each location

        # Structuring the prompt
        response_schemas = [
            ResponseSchema(name="commentary", description="news about the disaster"),
            ResponseSchema(name="date", description="date of the disaster"),
            ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        # Generate commentary for each location one by one
        responses = []
        for location_data in locations_data:
            location = location_data['location']
            question = f"The user is interested in finding disaster commentary for the location {location}. Find the latest news about the disasters along with the date and the location of the disaster, along with the source (link) of the news"
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template("Answer the user's question as best as possible.\n{format_instructions}\n{question}")  
                ],  
                input_variables=["question"],
                partial_variables={"format_instructions": format_instructions}
            )

            input = prompt.format_prompt(question=question)
            response = agent.run(input)
            # convert response to json
            response = output_parser.parse(response)
            # add location to response
            response['location'] = location
            # add latitude and longitude to response
            response['latitude'] = location_data['latitude']
            response['longitude'] = location_data['longitude']

            responses.append(response)

        with open('responses.json', 'w') as f:
            json.dump(responses, f)

    # 3. Return the data
    return jsonify(responses)
    

if __name__ == '__main__':
    app.run(debug=True, port= 3001)