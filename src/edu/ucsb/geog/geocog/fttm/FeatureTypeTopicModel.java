/*
 * (C) Copyright 2011, Benjamin Adams (badams at cs dot ucsb dot edu) 
 */
/*
 * FeatureTypeTopicModel is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option) any
 * later version.
 */
/*
 * FeatureTypeTopicModel is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 */
/*
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place, Suite 330, Boston, MA 02111-1307 USA
 */

/*
 * Created on September 25, 2011
 */
package edu.ucsb.geog.geocog.fttm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map;


public class FeatureTypeTopicModel {
  public static final byte FEATURE_TYPE = 0;
  public static final byte ABSTRACT = 1;
  public static final byte UNKNOWN = -1;
  
  String[] vocabulary;
  String[] featureTypeTerms;
  int F; // number of feature types
  int D; // number of documents
  int Tfeat; // number of feature type topics
  int Tabst; // number of abstract topics
  int T; // total number of topics
  int[] N; // number of words by document
  int V; // vocabulary size
  int numberOfDocuments;
  
  double alpha; // Dirichlet prior on abstract topics
  double betaFeature; // Dirichlet prior (word, feature type topic associations)
  double betaAbstract; // Dirichlet prior (word, abstract topic associations)
  double psi; // Dirichlet prior (feature type topic, feature associations)
  double gammaFeature; // Beta prior on feature type topics (symmetric beta)
  double gammaAbstract;
  
  int[][] w; // word assignment for document d, word i
  double[][] P; // feature type distributions D X F matrix
  int[][] f; // feature type assignment for document d, word i (-1 abstract topic)
  int[][] z; // topic assignments for document d, word i, feature type topics followed by abstract topics
  byte[][] x; // binary switch assignment for document d, word i (0 = feature type, 1 = abstract)
  
  int[][] n_w_abst_z; // number of times word w is assigned to abstract topic z
  int[][] n_d_abst_z; // number of times a word in document d is assigned to abstract topic z
  int[] n_d_abst; // number of times a word in document d is assigned to an abstract topic
  int[][] n_w_feat_z; // number of times the word w is assigned to feature type topic z
  int[][] n_d_feat_z; // number of times a word in document d is assigned to feature type topic z
  int[][] n_fz_f; // number of times feature type topic z is assigned to feature type f
  int[] n_fz_F; // number of times feature type topic z is assigned to a feature type
  int[] n_d_feat; // number of times a word in document d is assigned to a feature type topic
  int[][] n_d_f; // number of times a word in document d is assigned to feature type f  
  
  int[] n_z; // total number of words assigned to topic z

  
  /**
   * cumulative statistics of theta
   */
  double[][] theta_sum;  // cumulative statistics for theta (prob. of abstract topics)

  /**
   * cumulative statistics of phi
   */
  double[][] phi_feat_sum;  // cumulative statistics for phi feat
  double[][] phi_abst_sum;  // cumulative statistics for phi abst
  double[][] epsilon; // probabilities of feature type topics  given feature type f
  
  public FeatureTypeTopicModel(int[][] documents, double[][] featureTypeDistributions, String[] vocabulary, 
		                       String[] featureTypeTerms, 
		  					   int vocabularySize,
		                       int numberOfFeatureTypeTopics, int numberOfAbstractTopics, double alpha,
		                       double betaFeature, double betaAbstract, double psi, double gammaFeature,
		                       double gammaAbstract) {
	  w = documents;
	  V = vocabularySize;
	  this.vocabulary = vocabulary;
	  this.featureTypeTerms = featureTypeTerms;
	  P = featureTypeDistributions;
	  Tfeat = numberOfFeatureTypeTopics;
	  Tabst = numberOfAbstractTopics;
	  T = Tfeat + Tabst;
	  this.alpha = alpha;
	  this.betaFeature = betaFeature;
	  this.betaAbstract = betaAbstract;
	  this.psi = psi;
	  this.gammaFeature = gammaFeature;
	  this.gammaAbstract = gammaAbstract;
	  
	  numberOfDocuments = w.length;
	  if (P.length != numberOfDocuments) { 
		  throw new IndexOutOfBoundsException("Number of documents different in word and feature type distribution matrices."); 
	  }
	  F = P[0].length;
	  for (int i = 1; i < P.length; i++) {
		  if (P[i].length != F) {
			  throw new IndexOutOfBoundsException("All documents do not share same number of feature types");
		  }
	  }
	  N = new int[numberOfDocuments];
  }
  
  private void initialize() {
	  f = new int[numberOfDocuments][];
	  z = new int[numberOfDocuments][];
	  x = new byte[numberOfDocuments][];
	  
	  n_w_abst_z = new int[V][Tabst];
	  n_z = new int[Tfeat+Tabst];
	  n_d_abst_z = new int[numberOfDocuments][Tabst];
	  n_d_abst = new int[numberOfDocuments];
	  n_w_feat_z = new int[V][Tfeat];
	  n_fz_f = new int[Tfeat][F];
	  n_fz_F = new int[Tfeat];
	  n_d_feat = new int[numberOfDocuments];
	  n_d_f = new int[numberOfDocuments][F];
	  
	  for (int d = 0; d < numberOfDocuments; d++) {
		  int numWords = w[d].length;
		  z[d] = new int[numWords];
		  f[d] = new int[numWords];
		  x[d] = new byte[numWords];
		  for (int i = 0; i < numWords; i++) {
			  int topic = (int) (Math.random() * (Tfeat + Tabst));
			  z[d][i] = topic;
			  // number of instances of word i assigned to topic z
			  if (topic < Tfeat) { // it is feature type
				  n_w_feat_z[w[d][i]][topic]++;
				  n_z[topic]++;
				  n_d_feat[d]++;
				  
				  // initialize feature type prop. to features for doc.
				  double sample = Math.random();
				  double cumulative = 0.0;
				  for (int feature = 0; feature < F; feature++) {
					  cumulative = cumulative + P[d][feature];
					  if (sample < cumulative) {
						  f[d][i] = feature;
						  n_d_f[d][feature]++;
						  break;
					  }
				  }
				  //f[d][i] = (int) (Math.random() * F);
				  n_fz_f[topic][f[d][i]]++;
				  n_fz_F[topic]++;
				  
				  // initialize binary switch
				  x[d][i] = FEATURE_TYPE;
				  
			  } else { // it is abstract 
				  n_w_abst_z[w[d][i]][topic-Tfeat]++;
				  n_z[topic]++;
				  n_d_abst_z[d][topic-Tfeat]++;
				  n_d_abst[d]++;
				  
				  // initialize feature type
				  f[d][i] = -1;
				  
				  // initialize binary switch
				  x[d][i] = ABSTRACT;
			  } 
		  }
		  N[d] = numWords;
	  }
  }
  
  public void doSampling(int iterations) {
	  initialize();
	  
	  for (int k = 0; k < iterations; k++) {
		  long starttime = System.currentTimeMillis();
		  System.out.print("Iteration " + k);
		  for (int d = 0; d < w.length; d++) {
			  if (d % 1000 == 0) {
				  System.out.print(".");
			  }
			  //System.out.println(d);
		      int ab = 0;
		      int ft = 0;
			  for (int i = 0; i < w[d].length; i++) {
				  int word = w[d][i];
				  int topic = z[d][i];
				  int featureType = f[d][i];
				  byte binarySwitch = x[d][i];
				  
				  // remove topic, feature type (as appropriate) from count matrices
				  if (topic < Tfeat) {
					  n_w_feat_z[word][topic]--;
					  n_d_feat[d]--;
					  n_d_f[d][featureType]--;
					  n_fz_f[topic][featureType]--;
					  n_fz_F[topic]--;
				  } else {
					  n_w_abst_z[word][topic-Tfeat]--;
					  n_d_abst_z[d][topic-Tfeat]--;
					  n_d_abst[d]--;
				  }
				  n_z[topic]--;
				  // decrement word count from document
				  N[d]--;
				  
				  //TODO
				  // cumulative method multinomial sampling
				  double[] probs = new double[Tfeat*F + Tabst]; // total # of probabilities based on topics with features and abstract topics
				  // do feature type topics first
				  for (int t = 0; t < Tfeat; t++) { //TODO: WHY ARE psi and alpha duking it out here?
					  for (int f = 0; f < F; f++) { // prob. ft. topic t and feature f
						  double p_ratio = 0.0;
						  if (n_d_f[d][f] == 0 || n_d_feat[d] == 0) {
							  p_ratio = P[d][f];
						  } else {
						    p_ratio = (P[d][f]) / ((double)n_d_f[d][f] / (double)n_d_feat[d]);
						  }
						  probs[(t*F)+f] = (n_w_feat_z[word][t] + betaFeature) / (n_z[t] + V * betaFeature) *
						                         p_ratio *
						                         (n_fz_f[t][f] + psi) / (n_fz_F[t] + F * psi) *
						                         (n_d_feat[d] + gammaFeature);  // TODO: WHY IS it feature 7 is dominating?
					  }
				  }
				  for (int t = 0; t < Tabst; t++) { // prob. abstract topic t
					  probs[Tfeat*F + t] = (n_w_abst_z[word][t] + betaAbstract) / (n_z[Tfeat+t] + V * betaAbstract) *
					                       (n_d_abst_z[d][t] + alpha) / (n_d_abst[d] + Tabst*alpha) *
					                       (n_d_abst[d] + gammaAbstract);
				  }
				  
				  // cumulate multinomial
				  for (int kk = 1; kk < Tfeat * F + Tabst; kk++) {
					  probs[kk] += probs[kk-1];
				  }
				  
				  // scaled sample because of unnormalised p[]
			      double u = Math.random() * probs[Tfeat * F + Tabst - 1];
				  // TODO now do the sampling...
			      int new_topic = -1;
			      int new_feature = -1;

				  for (int kk = 0; kk < Tfeat * F + Tabst; kk++) {
					  if (u < probs[kk]) {
						  if (kk < Tfeat * F) { // it is a feature topic
							  new_topic = kk / F;
							  new_feature = kk % F;
							  //ft++;
						  } else {  // it is an abstract topic
							  new_topic = Tfeat + (kk - Tfeat * F);
							  new_feature = -1;
							  //ab++;
						  }
						  break;
					  }
				  }
				  // add newly estimated topic, feature to count values
				  if (new_topic < Tfeat) { // it is a feature type topic
					  n_w_feat_z[word][new_topic]++;
					  n_d_feat[d]++;
					  n_d_f[d][new_feature]++;
					  n_fz_f[new_topic][new_feature]++;
					  n_fz_F[new_topic]++;
				  } else {
					  n_w_abst_z[word][new_topic-Tfeat]++;
					  n_d_abst_z[d][new_topic-Tfeat]++;
					  n_d_abst[d]++;
				  }
				  n_z[new_topic]++;
				  N[d]++;
				  z[d][i] = new_topic;
				  f[d][i] = new_feature;
				  x[d][i] = (new_feature < Tfeat) ? FEATURE_TYPE : ABSTRACT;
			  }
			  //System.out.println(ab+","+ft);

		  }
		  long endtime = System.currentTimeMillis();
		  System.out.print(((endtime-starttime)/1000));
		  System.out.println();
		  if ((k > 0) && (k % 50 == 0)) {
			  System.out.println("======================");
			  System.out.println("Report iteration "+k);
			  System.out.println("======================");
			  double[][] phi = getPhi();
			  int[][] topWords = getTopWords(phi, 20);
			  for (int i = 0; i < topWords.length; i++) {
				  String outs = "topic " + i + ": ";
				  for (int j = 0; j < topWords[i].length; j++) {
					  outs = outs + vocabulary[topWords[i][j]] + " ";
				  }
				  System.out.println(outs);
			  }
			  double[][] epsilon = getEpsilon();
			  System.out.println();
			  System.out.println(epsilonPrintOut(epsilon));
		  }
	  }
	  
	  System.out.println("Final report:");
	  double[][] phi = getPhi();
	  int[][] topWords = getTopWords(phi, 20);
	  for (int i = 0; i < topWords.length; i++) {
		  String outs = "topic " + i + ": ";
		  for (int j = 0; j < topWords[i].length; j++) {
			  outs = outs + vocabulary[topWords[i][j]] + " ";
		  }
		  System.out.println(outs);
	  }
	  double[][] epsilon = getEpsilon();
	  System.out.println();
	  System.out.println(epsilonPrintOut(epsilon));
	  System.out.println("======================");
	  //System.out.println("Counts:");
	  //System.out.println(epsilonPrintOutInt(n_fz_f));
	  //System.out.println("======================");
	  System.out.println("Phi:");
	  //System.out.println(printOut(getPhi()));
	  printPrintOut(getPhi());
	  System.out.println("Theta:");
	  //System.out.println(printOut(getThetaAbstract()));
	  printPrintOut(getThetaAbstract());
	  System.out.println("Done sampling.");
  }
  
  public double[][] getPhi() {
	  double[][] phi = new double[T][V];
	  for (int t = 0; t < Tfeat; t++) { // do feature topics
		  for (int w = 0; w < V; w++) {
			  phi[t][w] = (n_w_feat_z[w][t] + betaFeature) / (n_z[t] + V * betaFeature);
		  }
	  }
	  for (int t = Tfeat; t < T; t++) { // do abstract topics
		  for (int w = 0; w < V; w++) {
			  phi[t][w] = (n_w_abst_z[w][t-Tfeat] + betaAbstract) / (n_z[t] + V * betaAbstract);
		  }
	  }
	  return phi;
  }
  
  public String printOut(double[][] data) {
	  String outs = "";
	  for (int i = 0; i < data.length; i++) {
		  outs = outs + i + "\t";
		  for (int j = 0; j < data[i].length; j++) {
			  outs = outs + data[i][j] + " ";
		  }
		  outs = outs + "\n";
	  }
	  return outs;
  }
  
  public void printPrintOut(double[][] data) {
	  for (int i = 0; i < data.length; i++) {
		  System.out.print(i);
		  System.out.print("\t");
		  for (int j = 0; j < data[i].length; j++) {
			  System.out.print(data[i][j]);
			  System.out.print(" ");
		  }
		  System.out.println();
	  }
  }
  
  public double[][] getThetaAbstract() {
	  double[][] theta = new double[numberOfDocuments][Tabst];
	  for (int d = 0; d < numberOfDocuments; d++) {
		  for (int t = 0; t < T; t++) {
			  if (t >= Tfeat) {
				  theta[d][t-Tfeat] = (n_d_abst_z[d][t-Tfeat] + alpha) / (n_d_abst[d] + Tabst*alpha);
			  }
		  }
	  }
	  return theta;
  }
  
  public double[][] getEpsilon() {
	  double[][] epsilon = new double[F][Tfeat];
	  for (int f = 0; f < F; f++) {
		  double sum = 0.0;
		  for (int t = 0; t < Tfeat; t++) {
			  sum += n_fz_f[t][f];
		  }
		  for (int t = 0; t < Tfeat; t++) {
			  //epsilon[f][t] = (n_fz_f[t][f] + psi) / (n_fz_F[t] + F * psi);
			  if (sum > 0.0) {
				  epsilon[f][t] = (float)n_fz_f[t][f] / sum;
			  } else {
				  epsilon[f][t] = 0.0;
			  }
		  }
	  }
	  return epsilon;
  }
  
  public String epsilonPrintOut(double[][] epsilon) {
	  //String[] featureType = {"Stream", "Lake", "Mountain", "Valley", "Beach", "Building", "Park"};
	  String outs = "";
	  for (int f = 0; f < F; f++) {
		  outs = outs + featureTypeTerms[f] + "\t";
		  for (int t = 0; t < Tfeat; t++) {
			  outs = outs + epsilon[f][t] + " ";
		  }
		  outs = outs + "\n";
	  }
	  return outs;
  }
  
  public String epsilonPrintOutInt(int[][] epsilon) {
	  //String[] featureType = {"Stream", "Lake", "Mountain", "Valley", "Beach", "Building", "Park"};
	  String outs = "";
	  for (int f = 0; f < F; f++) {
		  outs = outs + featureTypeTerms[f] + "\t";
		  for (int t = 0; t < Tfeat; t++) {
			  outs = outs + epsilon[f][t] + " ";
		  }
		  outs = outs + "\n";
	  }
	  return outs;
  }
  
  public int[][] getTopWords(double[][] phi, int n) { //get top n words for each topic from phi
	  int[][] topWords = new int[T][n];
	  for (int t = 0; t < T; t++) {
		  Hashtable<Integer,Double> ht = new Hashtable<Integer,Double>(V*2);
		  for (int i = 0; i < V; i++) {
			  ht.put(i, phi[t][i]);
		  }
		  ArrayList al = new ArrayList(ht.entrySet());
		  //Collections.sort(al, new DoubleIntegerComparator());
		  Collections.sort(al, new Comparator(){
			  public int compare(Object obj1, Object obj2){
				  int result=0;
				  Map.Entry e1 = (Map.Entry)obj1 ;
				  Map.Entry e2 = (Map.Entry)obj2 ;//Sort based on values.

				  Double value1 = (Double)e1.getValue();
				  Double value2 = (Double)e2.getValue();

				  if(value1.compareTo(value2)==0) {
					  Integer key1=(Integer)e1.getKey();
					  Integer key2=(Integer)e2.getKey();

					  //Sort String in an alphabetical order
					  result=key1.compareTo(key2);
				  } else {
					  //Sort values in a descending order
					  result=value2.compareTo( value1 );
				  }

				  return result;
			  }
		  });
		  Iterator itr = al.iterator();
		  int cnt = 0;
		  while (cnt < n && itr.hasNext()) {
			  Map.Entry e = (Map.Entry)itr.next();
			  topWords[t][cnt] = (Integer)e.getKey();
			  cnt++;
		  }
	  }
	  return topWords;
  }
  /**
  static class DoubleIntegerComparator implements Comparator<Object>{

	  public int compare(Object obj1, Object obj2){
		  int result=0;
		  Map.Entry e1 = (Map.Entry)obj1 ;
		  Map.Entry e2 = (Map.Entry)obj2 ;//Sort based on values.

		  Double value1 = (Double)e1.getValue();
		  Double value2 = (Double)e2.getValue();

		  if(value1.compareTo(value2)==0) {
			  Integer key1=(Integer)e1.getKey();
			  Integer key2=(Integer)e2.getKey();

			  //Sort String in an alphabetical order
			  result=key1.compareTo(key2);
		  } else {
			  //Sort values in a descending order
			  result=value2.compareTo( value1 );
		  }

		  return result;
	  }

  }
  **/
  public static void main(String[] args) {
	  String fn = "sampledata/fttm_input_500_5_us.txt";
	  String fn_words = "sampledata/unique_words_merged_us_docs_5_500.txt";
	  String fn_feat_types = "sampledata/max_per_feature_type_no_4_after_500.txt";
	  if (args.length > 0) {
		  fn = args[0];
	  }
	  if (args.length > 1) {
		  fn_words = args[1];
	  }
	  if (args.length > 2) {
		  fn_feat_types = args[2];
	  }
	  System.out.println(fn);
	  System.out.println(fn_words);
	  System.out.println(fn_feat_types);
	  
	  //# feature topics, # abst topics, alpha (abst), beta(feat), beta(abst), psi(f.t. -> f), gamma(feature), gamma(abst)
	  //10, 50, 0.8, 0.1, 0.1, 50.0 / numFeatures, 0.01, 2.0
	  int numFeatureTopics = 10;
	  int numAbstractTopics = 10;
	  double alpha = 0.8;
	  double betaFeature = 0.1;
	  double betaAbstract = 0.1;
	  double psi = 50. / 85.;
	  double gammaFeature = 0.01;
	  double gammaAbstract = 2.0;
	  int numberOfIterations = 1;
	  if (args.length > 3) {
		  numFeatureTopics = Integer.parseInt(args[3]);  
	  }
	  if (args.length > 4) {
		  numAbstractTopics = Integer.parseInt(args[4]);
	  }
	  if (args.length > 5) {
		  alpha = Double.parseDouble(args[5]);
	  }
	  if (args.length > 6) {
		  betaFeature = Double.parseDouble(args[6]);
	  }
	  if (args.length > 7) {
		  betaAbstract = Double.parseDouble(args[7]);
	  }
	  if (args.length > 8) {
		  psi = Double.parseDouble(args[8]);
	  }
	  if (args.length > 9) {
		  gammaFeature = Double.parseDouble(args[9]);
	  }
	  if (args.length > 10) {
		  gammaAbstract = Double.parseDouble(args[10]);
	  }
	  if (args.length > 11) {
		  numberOfIterations = Integer.parseInt(args[11]);
	  }
	  System.out.print("Number of feature topics:  ");
	  System.out.println(numFeatureTopics);
	  System.out.print("Number of abstract topics: ");	  
	  System.out.println(numAbstractTopics);
	  System.out.print("Alpha: ");
	  System.out.println(alpha);
	  System.out.print("Beta feature:  ");
	  System.out.println(betaFeature);
	  System.out.print("Beta abstract: ");
	  System.out.println(betaAbstract);
	  System.out.print("Psi: ");
	  System.out.println(psi);
	  System.out.print("Gamma feature:  ");
	  System.out.println(gammaFeature);
	  System.out.print("Gamma abstract: ");
	  System.out.println(gammaAbstract);
	  System.out.print("Number of iterations: ");
	  System.out.println(numberOfIterations);
	  
      long heapSize = Runtime.getRuntime().totalMemory();
      //Print the jvm heap size.
      System.out.println("Heap Size = " + heapSize);
	  int lineNumber = 0;
	  int numDocs = 0;
	  int numFeatures = 0;
	  int vocabSize = 0;
	  int[] docIds = null;
	  double[][] feature_p = null;
	  int[][] words = null;
	  try {
		  BufferedReader in = new BufferedReader(new FileReader(fn));
		  String str;
		  int documentCount = 0;
		  while ((str = in.readLine()) != null) { // read in the data file
			  if (lineNumber == 0) { // header info
				  String[] stats = str.split(" ");
				  numDocs = Integer.parseInt(stats[0]);
				  numFeatures = Integer.parseInt(stats[1]);
				  vocabSize = Integer.parseInt(stats[2]);
				  docIds = new int[numDocs];
				  feature_p = new double[numDocs][numFeatures];
				  words = new int[numDocs][];
			  } else {
				  if (lineNumber % 2 == 1) { // feature pcts
					  String[] tokenz = str.split("\t");
					  docIds[documentCount] = Integer.parseInt(tokenz[0]);
					  String[] tokens = tokenz[1].split(" ");
					  //String[] tokens = str.split(" ");
					  //System.out.println(tokens.length);
					  //docIds[documentCount] = Integer.parseInt(tokens[0]);
					  //for (int j = 1; j <= numFeatures ; j++) {
						//  feature_p[documentCount][j-1] = Double.parseDouble(tokens[j]);
					  //}
					  for (int j = 0; j < numFeatures ; j++) {
						  feature_p[documentCount][j] = Double.parseDouble(tokens[j]);
					  }
				  } else { // words
					  String[] tokenz = str.split("\t");
					  //String[] tokens = str.split(" ");
					  String[] tokens = tokenz[2].split(" ");
					  words[documentCount] = new int[tokens.length];
					  for (int j = 0; j < tokens.length; j++) {
						  words[documentCount][j] = Integer.parseInt(tokens[j]);
					  }
					  //
					  documentCount++;
				  }
			  }
			  lineNumber++;
			  //System.out.println(lineNumber);
		  }
		  in.close();
	  } catch (IOException e) { System.err.println(e); }
	  String[] vocabulary = null;
	  try{
		  BufferedReader in = new BufferedReader(new FileReader(fn_words));
		  String str;
		  vocabulary = new String[vocabSize];
		  int cnt = 0;
		  while ((str = in.readLine()) != null) { // read in the data file
			  //String[] line = str.split(" ");
			  //String word = line[1].trim();
			  String word = str.trim();
			  vocabulary[cnt] = word;
			  cnt++;
		  }
	  } catch (IOException e) { System.err.println(e); }
	  
	  String[] featureTypes = null;
	  try{
		  BufferedReader in = new BufferedReader(new FileReader(fn_feat_types));
		  String str;
		  featureTypes = new String[numFeatures];
		  int cnt = 0;
		  while ((str = in.readLine()) != null) { // read in the data file
			  String[] line = str.split("\t");
			  String word = line[0].trim();
			  //String word = str.trim();
			  featureTypes[cnt] = word;
			  cnt++;
		  }
	  } catch (IOException e) { System.err.println(e); }
	  
	  // W, P, V, # feature topics, # abst topics, alpha (abst), beta(feat), beta(abst), psi(f.t. -> f), gamma(feature), gamma(abst)
	  //FeatureTypeTopicModel fttm = new FeatureTypeTopicModel(words, feature_p, vocabulary, vocabSize, 10, 30, 1.0, 0.1, 0.1, 2.0, 1.0, 500.0);
	  //FeatureTypeTopicModel fttm = new FeatureTypeTopicModel(words, feature_p, vocabulary, vocabSize, 25, 50, 1.0, 0.01, 0.1, 0.3, 1.0, 1000.0);
	  FeatureTypeTopicModel fttm = new FeatureTypeTopicModel(words, feature_p, vocabulary, featureTypes, vocabSize, numFeatureTopics, numAbstractTopics, alpha, betaFeature, betaAbstract, psi, gammaFeature, gammaAbstract);

	  fttm.doSampling(numberOfIterations);
  }
}
