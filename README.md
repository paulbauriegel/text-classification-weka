# Text Classification with Weka
This repo contains some test for a Text Classification based on models trained with the Weka Explorer. The code however is does not work properly at the moment.

Few comments on how to setup this repo:
1. Install Maven (https://maven.apache.org/)
2. Install the snowball-stemmers dependency into local maven Repository:
	2.1. Download newest snowball-stemmers weka package or install using the jar in the lib/ folder
	2.2. `mvn install:install-file -Dfile=snowball-stemmers.jar -DgroupId=org.tartarus.snowball -DartifactId=snowball-stemmers -Dversion=1.0.2 -Dpackaging=jar`
3. Build the workspace dependencies
4. Run de.qaass.classifier.TextClassTest