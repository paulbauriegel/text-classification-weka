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
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.SnowballStemmer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 * Test Text Classification with MultinomialNB
 */
public class TextClassTest {
	public static void main(String[] args) {
		try {
			Classifier docCls = loadClassifier("data/corn.model");
			Instances baseVectorData = getDataFromFile("data/ReutersCorn-train.arff");
			System.out.println(baseVectorData.get(0));
			Instances newData = buildInstance(
					"FURTHER ARGENTINE COARSE GRAIN LOSSES FEARED Argentine grain producers adjusted\\ntheir yield estimates for the 1986/87 coarse grain crop\\ndownward in the week to yesterday after the heavy rains at the\\nend of March and beginning of April, trade sources said.\\n    They said sunflower, maize and sorghum production estimates\\nhad been reduced despite some later warm, dry weather, which\\nhas allowed a return to harvesting in some areas.\\n    However, as showers fell intermittently after last weekend,\\nproducers feared another spell of prolonged and intense rain\\ncould cause more damage to crops already badly hit this season.\\n    Rains in the middle of last week reached an average of 27\\nmillimetres in parts of Buenos Aires province, 83 mm in\\nCordoba, 41 in Santa Fe, 50 in Entre Rios and Misiones, 95 in\\nCorrientes, eight in Chaco and 35 in Formosa.\\n    There was no rainfall in the same period in La Pampa.\\n    Producers feared continued damp conditions could produce\\nrotting and lead to still lower yield estimates for all the\\ncrops, including soybean.\\n    However, as the lands began drying later in the week\\nharvesting advanced considerably, reaching between 36 and 40\\npct of the area sown in the case of sunflower.\\n    Deterioration of the sunflower crop evident in harvested\\nmaterial in Cordoba, La Pampa and Buenos Aires forced yield\\nestimates per hectare to be adjusted down again.\\n    The season\\'s sunflowerseed production is now forecast at\\n2.1 mln to 2.3 mln tonnes, against 2.2 mln to 2.4 mln forecast\\nlast week and down 43.9 to 48.8 pct on the 1985/86 record of\\n4.1 mln.\\n    Area sown to sunflowers was two to 2.2 mln hectares, 29.9\\nto 36.3 pct below the record 3.14 mln hectares last season.\\n    Maize harvesting has also reached 36 to 40 pct of the area\\nsown. It is near completion in Cordoba and Santa Fe and will\\nbegin in La Pampa and southern Buenos Aires later in April.\\n    Production estimates for maize were down from last week at\\n9.5 mln to 9.8 mln tonnes, against 9.6 mln to 9.9 mln estimated\\npreviously.\\n    This is 22.2 to 23.4 pct below the 12.4 mln to 12.6 mln\\ntonnes estimated by private sources for the 1985/86 crop and\\n21.9 to 25.8 pct down on the official figure of 12.8 mln\\ntonnes.\\n    Maize was sown on 3.58 mln to 3.78 mln hectares, two to\\nseven pct down on last season\\'s 3.85 mln.\\n    Sorghum was harvested on 23 to 25 pct of the area sown in\\nCordoba, Santa Fe and Chaco. Harvest will start in La Pampa and\\nBuenos Aires in mid-April.\\n    The total area sown was 1.23 mln to 1.30 mln hectares, 10.3\\nto 15.2 pct down on the 1.45 mln sown last season.\\n    The new forecast for the sorghum crop is 2.9 mln to 3.2 mln\\ntonnes compared with three mln to 3.3 mln forecast last week,\\nand is 23.8 to 29.3 pct down on last season\\'s 4.1 mln to 4.2\\nmln tonne crop.\\n    The soybean crop for this season was not adjusted,\\nremaining at a record 7.5 mln to 7.7 mln tonnes, up 4.2 to 5.5\\npct on the 7.2 mln to 7.3 mln estimated by private sources for\\n1985/86 and 5.6 to 8.5 pct higher than the official figure of\\n7.1 mln.\\n    The area sown to soybeans this season was a record 3.7 mln\\nto 3.8 mln hectares, 10.8 to 13.8 pct up on the record 3.34 mln\\nsown in 1985/86.\\n    The soybean crop is showing excessive moisture in some\\nareas and producers fear they may discover more damage. Some\\nexperimental harvesting was carried out in Santa Fe on areas\\nmaking up only about one pct of the total crop but details on\\nthis were not available.\\n    Preparation of the fields for the 1987/88 wheat crop, which\\nwill be sown between May and August or September, has so far\\nnot been as intense as in previous years.\\n Reuter\\n&#3;",
					"1");
			// Instances newData = getDataFromFile("data/ReutersCorn-proof.arff");
			System.out.println(newData.get(0));
			Instances filtered = convertToVecor(newData, newData);
			Instances filtered2 = convertToVecor(baseVectorData, baseVectorData);
			System.out.println(filtered);
			System.out.println(filtered2.get(0));
			double result = docCls.classifyInstance(filtered.get(0));
			String label = filtered.classAttribute().value((int) result);
			System.out.println(result + "-" + label);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static Instances buildInstance(String inputString, String inputClass) {
		ArrayList<Attribute> atts;
		ArrayList<String> attVals;
		Instances data;
		double[] vals;

		// 1. set up attributes
		atts = new ArrayList<Attribute>();
		// - string
		atts.add(new Attribute("Text", (ArrayList<String>) null));
		// - nominal
		attVals = new ArrayList<String>(Arrays.asList("0", "1"));
		atts.add(new Attribute("class-att", attVals));

		// 2. create Instances object
		data = new Instances("TestData", atts, 0);

		// 3. fill with data
		vals = new double[data.numAttributes()];
		// - nominal
		vals[1] = attVals.indexOf(inputClass);
		// - string
		vals[0] = data.attribute(0).addStringValue(inputString);
		// add
		data.add(new DenseInstance(1.0, vals));

		// 4. output data
		return data;
	}

	public static Instances getDataFromFile(String path) throws Exception {

		DataSource source = new DataSource(path);
		Instances data = source.getDataSet();

		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes() - 1);
		}

		return data;
	}

	public static Instances convertToVecor(Instances baseVectorData, Instances newData) throws Exception {
		StringToWordVector stv = new StringToWordVector();
		stv.setInputFormat(baseVectorData);
		stv.setIDFTransform(true);
		stv.setTFTransform(true);
		stv.setLowerCaseTokens(true);
		SnowballStemmer stemmer = new SnowballStemmer();
		stv.setStemmer(stemmer);
		stv.setDoNotOperateOnPerClassBasis(true);
		Instances filtered = Filter.useFilter(newData, stv);
		return filtered;
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
