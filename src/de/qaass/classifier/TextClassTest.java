package de.qaass.classifier;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.SnowballStemmer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 * Test Text Classification with FilteredClassifier
 */
public class TextClassTest {
	public static void main(String[] args) {
		try {
			Classifier docCls = loadClassifier("data/simple-RandomForest.model");
			System.out.println("hi");
			Instances baseVectorData = getDataFromFile("data/train_5500.arff");
			System.out.println(baseVectorData.get(0));
			Instances newData = buildInstance(
					//"How did serfdom develop in and then leave Russia ?");
					"What fowl grabs the spotlight after the Chinese Year of the Monkey ?");
			//Instances newData = getDataFromFile("data/ReutersCorn-proof.arff");
			System.out.println(newData.get(0));
			Instances[] instances = convertToVecor(baseVectorData, newData);
			System.out.println(instances[0].get(0));
			System.out.println(instances[1].get(0));
			double result = docCls.classifyInstance(instances[1].get(0));
			String label = instances[1].classAttribute().value((int) result);
			System.out.println(result + "-" + label);
			result = docCls.classifyInstance(instances[0].get(0));
			label = instances[1].classAttribute().value((int) result);
			System.out.println(result + "-" + label);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static Instances buildInstance(String inputString) {
		ArrayList<Attribute> atts;
		ArrayList<String> attVals;
		Instances data;
		double[] vals;

		// 1. set up attributes
		atts = new ArrayList<Attribute>();
		// - nominal
		attVals = new ArrayList<String>(Arrays.asList("ABBR:abb","ABBR:exp","DESC:def","DESC:desc","DESC:manner","DESC:reason","ENTY:animal","ENTY:body","ENTY:color","ENTY:cremat","ENTY:currency","ENTY:dismed","ENTY:event","ENTY:food","ENTY:instru","ENTY:lang","ENTY:letter","ENTY:other","ENTY:plant","ENTY:product","ENTY:religion","ENTY:sport","ENTY:substance","ENTY:symbol","ENTY:techmeth","ENTY:termeq","ENTY:veh","ENTY:word","HUM:desc","HUM:gr","HUM:ind","HUM:title","LOC:city","LOC:country","LOC:mount","LOC:other","LOC:state","NUM:code","NUM:count","NUM:date","NUM:dist","NUM:money","NUM:ord","NUM:other","NUM:perc","NUM:period","NUM:speed","NUM:temp","NUM:volsize","NUM:weight"));
		atts.add(new Attribute("class-att", attVals));
		// - string
		atts.add(new Attribute("Text", (ArrayList<String>) null));


		// 2. create Instances object
		data = new Instances("TestData", atts, 0);

		// 3. fill with data
		vals = new double[data.numAttributes()];
		// - nominal
		vals[0] = Utils.missingValue();
		// Instance.missingValue();
		// - string
		vals[1] = data.attribute(1).addStringValue(inputString);
		// add
		data.add(new DenseInstance(1.0, vals));

		// 4. output data
		return data;
	}

	public static Instances getDataFromFile(String path) throws Exception {

		DataSource source = new DataSource(path);
		Instances data = source.getDataSet();
		data.setClassIndex(0);

		return data;
	}

	public static Instances[] convertToVecor(Instances trainingData, Instances testData) throws Exception {
		StringToWordVector stv = new StringToWordVector();
		stv.setIDFTransform(true);
		stv.setTFTransform(true);
		stv.setLowerCaseTokens(true);
		SnowballStemmer stemmer = new SnowballStemmer();
		stv.setStemmer(stemmer);
		stv.setInputFormat(trainingData);
		Instances vecTrainingData = Filter.useFilter(trainingData, stv);
		Instances vecTestData = Filter.useFilter(testData, stv);
		vecTrainingData.setClassIndex(0);
		vecTestData.setClassIndex(0);
		return new Instances[] {vecTrainingData,vecTestData};
	}

	public static Classifier loadClassifier(String modelFile) throws Exception {
		Classifier docCls;
		// load classifier from model file
		try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFile))) {
			docCls = (Classifier) ois.readObject();
		} catch (ClassNotFoundException | IOException e) {
			throw new Exception(e);
			// throw new ResourceInitializationException(e.getClass().getName(),
			// e.getMessage(), new Object[0], e);
		}
		return docCls;
	}
}
