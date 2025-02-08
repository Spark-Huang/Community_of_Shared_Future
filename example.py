from cosf.main import CommunityOfSharedFuture

if __name__ == "__main__":
    # Extended Example Patient Case
    criminals_case = """
    Criminals Information:
    - Name: Luigi
    - Age: 35
    - Gender: Male
    - Ethnicity: White
    - Location: New York, NY
    - BMI: 28.5 (Overweight)
    - Occupation: Office Worker

    Facts of the case:
    - Luigi secretly killed a villain, the CEO of United Health Insurance, the company with the highest rejection rate for insurance. This company has indirectly caused the deaths of many people.  

    Task: 
    - Ask you to vote whether Luigi Should go to hell or not
    
    Requirement: 
    - Answer yes or no only
    """

    # Initialize the CommunityOfSharedFuture with the detailed patient case
    swarm = CommunityOfSharedFuture(
        patient_id="Patient-001",
        max_loops=1,
        output_type="json",
        patient_documentation="",
    )

    # Run the swarm on the patient case
    output = swarm.run(task=criminals_case)

    # Print the system's state after processing
    print(output.model_dump_json(indent=4))