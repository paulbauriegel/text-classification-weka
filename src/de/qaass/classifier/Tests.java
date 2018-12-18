package de.qaass.classifier;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stemmers.Stemmer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.functions.LibLINEAR;

public class Tests {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			Instances trainingData = getDataFromFile("data/ReutersCorn-train.arff");
		    StringToWordVector filter = new StringToWordVector();    
		    filter.setWordsToKeep(1000000);
		    filter.setIDFTransform(true);
		    filter.setTFTransform(true);
		    filter.setLowerCaseTokens(true);
		    filter.setOutputWordCounts(true);
		    filter.setNormalizeDocLength(new SelectedTag(StringToWordVector.FILTER_NORMALIZE_ALL,StringToWordVector.TAGS_FILTER));
	        Stemmer s = new /*Iterated*/LovinsStemmer();
	        filter.setStemmer(s);
		    filter.setInputFormat(trainingData);
		    Instances trainingData2 = Filter.useFilter(trainingData, filter);
		    Classifier cls = null;
	        LibLINEAR liblinear = new LibLINEAR();
	        liblinear.setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
	        liblinear.setProbabilityEstimates(true);
	        // liblinear.setBias(1); // default value
	        cls = liblinear;
	        cls.buildClassifier(trainingData2);
	        Instances newData = getDataFromFile("data/ReutersCorn-proof.arff");
	        Instances testingData = Filter.useFilter(newData, filter);
	        for (int j = 0; j < testingData.numInstances(); j++) {
	            double res = cls.classifyInstance(testingData.get(j));
	            System.out.println(res);
	         }
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	private static Instances getDataFromFile(String path) throws Exception{

	    DataSource source = new DataSource(path);
	    Instances data = source.getDataSet();
	    
	    if (data.classIndex() == -1)
	        data.setClassIndex(data.numAttributes()-1);
	    
	    return data;    
	}

}
