# Sentiment-Analysis-Bucketing
*Categorizing Product reviews in appropriate categories.*

This solution enables us to classify the raw reviews into categories according to a pre-decided bucketing scheme.

The process is broadly divided into two steps:

**Raw Reviews**

   **⇓ 1**

**Entity Sentiment Table**

   **⇓ 2**	

**Final Output with Categories and Sub-Categories**
 
- **Step 1:** Making Use of GCP Natural Language API for Entity Sentiment Analysis and Sentiment Analysis.
    - Entity Sentiment Analysis gets entities and their corresponding salience score, sentiment score, and sentiment magnitude.
    - Sentiment Analysis gets the overall sentiment score of the reviews.
    - Uploading the resultant data to BigQuery.

- **Step 2:** Determining the respective categories of each review. 
    - Firstly, a general Level 1 dictionary is used to map the entities directly to their Level 1 Categories.
    - Second, a more general Level 2 dictionary is used along with the GloVe representation of words (100 dim). Closeness to cluster centroid of each category is used to determine the Level 1 category for remaining reviews.
    - A threshold is fixed based on the within-category deviation of distances of words in each cluster. The threshold metric can be min distance, max distance, standard deviation, or mean absolute deviation.
    - Next, for the reviews where no entities were detected, POS tagging is used to extract nouns, adjectives, and verbs from the review. A combined vector representation of the extracted words is used to determine the Category by again measuring the closeness.
    - Finally, Upload the resultant data to BigQuery Table.
 
**Fields in Raw Reviews,**
- Review_ID: Unique Review ID
- Comment: Review
- View: Title / Description
- Date: Date of Review
- Platform: Platform, ex: Amazon, Flipkart, etc.
- Product_Category: Type of Product, ex: fan, grinder, sewing machine, etc.
 
**Fields in Sentiment Analysis Table,**
- Review_ID: Unique Review ID		
- Reviews: Review			
- Overall_Sentiment_Score: Overall sentiment score of review			
- Overall_Sentiment_Magnitude: Overall sentiment magnitude			
- View: Title / Description
- Product_Category: Type of Product			
- Entity: Entity		
- Salience: Salience Score corresponding to entity			
- Sentiment_Score: Sentiment Score corresponding to entity
- Sentiment_Magnitude: Sentiment Magnitude corresponding to entity
 
**Fields in Final Output Table,**
(Individual tables are created for Each product type in Product_Category)
- Review_ID: Unique Review ID		
- Reviews: Review			
- Overall_Sentiment_Score: Overall sentiment score of review			
- Overall_Sentiment_Magnitude: Overall sentiment magnitude			
- View: Title / Description
- Product_Category: Type of Product			
- Entity: Entity		
- Salience: Salience Score corresponding to entity			
- Sentiment_Score: Sentiment Score corresponding to entity
- Sentiment_Magnitude: Sentiment Magnitude corresponding to entity
- Category: Level 1 Category estimated by the process.
 
 
This process makes us of a configuration file, a google sheet, sentiment_params.gsheet which is accessed and modified using Google Sheets API. link to g-sheet
- Task ID: Unique task ID.
- Client: Client / Project Name.
- Flag: 1 to run, 0 to not run for a give Task ID.
- File Type: File type of Mapping Sheet file.
- Raw Reviews Key File: Credential Json File to access the project.
- Raw Reviews Input: Raw Review table path.
- Sentiment Analysis Key File: Credential Json File to access the project.
- Sentiment Analysis Project ID: Project ID.
- Sentiment Analysis Dataset: Dataset.
- Sentiment Analysis Table: Table name.
- NL API Key File: Key file to use for NL API.
- Mapping Sheet: Mapping sheet ID.
- Output Key File: Credential Json File to access the project.
- Output Project ID: Project ID.
- Output Dataset: Dataset.
- Output Table Suffix: Output Table Suffix. 
- From Email: In case there is an error in uploading the files, an email is sent from this email id.
- From Email Password: Password for From_Email.
- To Email: In case there is an error in uploading the files, an email is sent to this email id.
