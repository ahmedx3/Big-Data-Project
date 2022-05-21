import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FileSystem;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.io.*;
import java.util.Iterator;
import java.util.HashMap;
import java.util.Map.Entry;



public class Knn {

  public static class DistaceCalculateMapper
       extends Mapper<LongWritable, Text, Text, Text>{

    // Static variables
    public static int numoffeatures;
    public static Float[] testSamplefeatures;
    public static ArrayList<String> samplesDistances = new ArrayList<String>();

    // Euclidean distance between features
    public static float euclideandist(Float[] trainSampleFeatures, int n)
    {
      // Init distance
      float distance = 0;

      // Loop over the elements and caluclate the euclidean distance between each two features
      for(int i = 0; i < n;i++)
      {
        distance += Math.pow(testSamplefeatures[i] - trainSampleFeatures[i], 2);
      }
      distance = (float)Math.sqrt(distance);

      // Return distance
      return distance;
    }

    // Setup function that is run once at the beggining
    public void setup(Context context) throws IOException, InterruptedException
    {
      // Set number of features taken from main
      // Minus 1 because last element is the class label
      numoffeatures = context.getConfiguration().getInt("numoffeatures",1) - 1;

      // Allocate array for the test features
      testSamplefeatures = new Float[numoffeatures];

      // Set the test features one by one
      for(int j=0;j<numoffeatures;j++)
      {
        testSamplefeatures[j] = context.getConfiguration().getFloat("feature" + j, 0);
      }
    }

    // Map function
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException
    {
      // Parse the features of the train sample
      String[] stringTrainFeatures = value.toString().split(",");
      Float[] trainFeatures = new Float[numoffeatures];
      for(int i = 0; i < numoffeatures; i++)
      {
        trainFeatures[i] = Float.parseFloat(stringTrainFeatures[i]);
      }

      // Get class label from data
      String classlabel;
      if (trainFeatures[numoffeatures - 1] == 1) {
        classlabel = "yes";
      } else {
        classlabel = "no";
      }

      // distance between sample and test point
      float distance = euclideandist(trainFeatures, numoffeatures);

      // Record the distance and label as a string
      samplesDistances.add(String.valueOf(distance) + "-" + classlabel);
    }

    // Cleanup function that is run once at the end
    public void cleanup(Context context) throws IOException, InterruptedException
    {
      // Sort the distances
      Collections.sort(samplesDistances);

      // Get the chosen k from main
      int chosenK = context.getConfiguration().getInt("chosenK", 5);

      // Send Closest K neighbours to reducer
      String[] closestSamples = new String[chosenK];
      String sample;

      for(int i = 0; i < chosenK; i++)
      {
        // Send same key for all to have them under a single reducer
        sample = samplesDistances.get(i);
        context.write(new Text("key"), new Text(sample));
      }
    }
  }

  public static class AggregatorReducer
       extends Reducer<Text, Text, Text, Text> {

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException
    {
      // Create string array of Neighbours
      ArrayList<String> neighbours = new ArrayList<String>();

      // Fill the string array from the values
      for (Text val : values) {
        context.write(new Text("Point"), val);
        neighbours.add(val.toString());
      }

      // Sort the distances
      Collections.sort(neighbours);

      // Get the chosen k from main
      int chosenK = context.getConfiguration().getInt("chosenK", 5);

      // Calculate max class
      int yes_count = 0;
      int no_count = 0;

      for(int i = 0; i < chosenK; i++)
      {
        // Get label
        String sample = neighbours.get(i);
        String label = sample.split("-")[1];

        // Add the label count
        if (label == "yes") {
          yes_count += 1;
        } else {
          no_count += 1;
        }
      }

      // Check which count is higher and set the label
      String predictedLabel;
      if (yes_count >= no_count) {
        predictedLabel = "yes";
      } else {
        predictedLabel = "no";
      }

      // Get true label
      String trueLabel;
      int labelNumber = 25; // Features are 25 and the label is the 26th which is index 25
      float trueLabelInt = context.getConfiguration().getFloat("feature" + labelNumber, -1);
      if (trueLabelInt == 1) {
        trueLabel = "yes";
      } else if (trueLabelInt == 0) {
        trueLabel = "no";
      } else {
        trueLabel = "not given";
      }

      // Write the value to output
      context.write(null, new Text("Predicted label : " + predictedLabel + "   " + "True Label : " + trueLabel));
    }
  }

  public static void main(String[] args) throws Exception {
    
    // Create configuration 
    Configuration conf = new Configuration();

    // Read test case file
    FileSystem hdfs = FileSystem.get(conf);
		BufferedReader br = new BufferedReader(new InputStreamReader(hdfs.open(new Path(args[0]))));
		
    // Parse file and set the features and K
    // Read K
    String line = br.readLine();
    int chosenK = Integer.parseInt(line);
    conf.setInt("chosenK", chosenK);
    
    // Read the features
    line = br.readLine();
    String[] features = line.split(",");
    int numoffeatures = features.length;

    for(int i = 0; i < numoffeatures; i++) {
      conf.setFloat("feature" + i, Float.parseFloat(features[i]));
    }
    
    // Close the file 
		br.close();
		hdfs.close();
		conf.setInt("numoffeatures", numoffeatures);

    // Start the map reduce job
    Job job = Job.getInstance(conf, "Knn");
    job.setJarByClass(Knn.class);
    job.setMapperClass(DistaceCalculateMapper.class);
    job.setReducerClass(AggregatorReducer.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(Text.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    FileInputFormat.addInputPath(job, new Path(args[1]));
    FileOutputFormat.setOutputPath(job, new Path(args[2]));
    job.waitForCompletion(true);
  }
}