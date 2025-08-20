# Orchestrator Test Setup

This guide explains how to set up the required Label Studio project and data for running the orchestrator integration tests.

## 1. Start Label Studio

You can use Docker Compose or your preferred method. Example (if you have a service defined):

```sh
docker-compose up -d labelstudio
```

Label Studio will be available at http://localhost:8080.

## 2. Create a Test Project in Label Studio

- Log in to Label Studio at http://localhost:8080
- Create a new project (e.g., "Test Project")
- Use the following labeling config:

```xml
<View>
  <Text name="text" value="$text"/>
  <View style="box-shadow: 2px 2px 5px #999;                padding: 20px; margin-top: 2em;                border-radius: 5px;">
    <Header value="Choose text sentiment"/>
    <Choices name="sentiment" toName="text" choice="multiple" showInLine="true">
      <Choice value="Positive"/>
      <Choice value="Negative"/>
      <Choice value="Neutral"/>
    </Choices>
    
  <Header value="Choose main topic"/>
  <Choices name="topic" toName="text" choice="single" showInLine="true">
    <Choice value="Politics"/>
    <Choice value="Sports"/>
    <Choice value="Technology"/>
    <Choice value="Entertainment"/>
    <Choice value="Business"/>
  </Choices>
    
  <Header value="What tags apply to this content? (Select all that apply)"/>
  <Choices name="tags" toName="text" choice="multiple" showInLine="true">
    <Choice value="Breaking News"/>
    <Choice value="Opinion"/>
    <Choice value="Analysis"/>
    <Choice value="Interview"/>
    <Choice value="Data-Driven"/>
    <Choice value="Personal Story"/>
  </Choices>
  
  <Header value="Select region"/>
  <Choices name="region" toName="text" choice="single" showInLine="true">
    <Choice value="Local"/>
    <Choice value="National"/>
    <Choice value="International"/>
  </Choices>
    
  </View>
</View>
```

## 3. Populate the Project with Test Data

You can use the following command to populate the project with data:

```sh
python -m mc_classifier_pipeline.data_ingest --query "climate" --start 2025-06-01 --end 2025-06-30 --project_id 1
```

- Make sure the `project_id` matches the ID of your Label Studio project.
- This will fetch and upload data to Label Studio for testing.

## 4. Run the Orchestrator Tests

From the project root, run:

```sh
pytest -vs tests/test_orchestrator.py
```

## Notes
- Ensure your environment variables (e.g., Label Studio API key, URL) are set as required by your pipeline.
- You can adjust the query, date range, or project ID as needed for your test scenario.

---

**This setup ensures your orchestrator tests have the correct project configuration and data in Label Studio.**
