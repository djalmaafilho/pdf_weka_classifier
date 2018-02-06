import java.io.File;
import java.util.TreeMap;
import java.util.TreeSet;

public class Treiner {
	public static TreeMap<String, Integer> generalStatistics = new TreeMap<>();
	static TreeSet<String> stopwords;
	static TraineUtils traineUtils = TraineUtils.newInstance();
	
	
	public static void main(String[] args) {		
		stopwords = traineUtils.loadStopwords();
		traineUtils.clearDir(traineUtils.getOutputDir());		
		stractClassFromDir(traineUtils.getDataDir());		
		traineUtils.generateArff(traineUtils.getDataDir(), traineUtils.getOutputDir(), generalStatistics);
		traineUtils.generateModels();
	}
	
	static void stractClassFromDir(File dir){
		for(File f : dir.listFiles()) {
			if(!f.isDirectory()){
				String txt = traineUtils.extractFromPdf(f,stopwords);
				txt = traineUtils.removeStopwords(txt, stopwords);
				traineUtils.extractWordsStatistcs(generalStatistics,txt);
				traineUtils.saveTxtToFile(traineUtils.getOutputDir(), f.getParentFile().getName()+"_"+f.getName().replace(".pdf", ".txt"), txt);
			}else{
				stractClassFromDir(f);
			}
		}
	}
}