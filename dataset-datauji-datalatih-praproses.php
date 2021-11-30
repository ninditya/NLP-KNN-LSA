<?php
declare(strict_types=1);
namespace IFVABOTKNN;

require './vendor/autoload.php';

ini_set('memory_limit', '-1');

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\NumericStringConverter;

include './fungsi.php';
include "./fungsi_svd.php";

$connect = connectDB('ifva');

$stopWords = file_get_contents("term_stopwords_id.txt"); //Panggil File Stopwords 

$stemmerFactory = new \Sastrawi\Stemmer\StemmerFactory(); //Panggil Kelas Stemmer
$stemmer  = $stemmerFactory->createStemmer();

//IMPORT DATA SET CSV
$csv =  Labeled::fromIterator(new ColumnPicker(new CSV('dataset/datasetfix.csv', true), ["pertanyaan", "jawaban", "kategori"]))->apply(new NumericStringConverter());
// $csv =  Labeled::fromIterator(new ColumnPicker(new CSV('dataset/datasetcoba.csv', true), ["pertanyaan", "jawaban", "label"]))->apply(new NumericStringConverter());
// [$training, $testing] = $csv->randomize()->stratifiedSplit(0.8);

// ======================================================================================================================================== //

$dataset = $csv->getSamples();
$labeldataset = $csv->getTargets();
// GABUNG DATASET Dengan Label
// print_r("<hr><pre><h1>DATASET</h1><br>");
$i=0;
$arr_dataset = " ";
foreach ($dataset as $kunci => $value) {
	$arr_dataset = $arr_dataset . ' '. $value[0];
	$pertanyaan_dataset[$i] = $value[0];
	$jawaban_dataset[$i] = $value[1];
	$i++;
}
print_r("<pre>");
$praproses_dataset = praproses($arr_dataset);
for ($i=0; $i < count($praproses_dataset) ; $i++) { 
	for ($j=0; $j < count($praproses_dataset[$i]) ; $j++) {  
		$stemVal_dataset = $stemmer->stem($praproses_dataset[$i][$j]);
		$hasilStem_dataset[$i][] = $stemVal_dataset;
		$stemVal2_dataset = $stemmer->stem($praproses_dataset[$i][$j]);
		// $hasilStem2_dataset[$no++] = $stemVal_dataset;
		// $term_dataset[$n++] = $praproses_dataset[$i][$j];
	}
	$hasilStemming_dataset[$i] = implode(" ", $hasilStem_dataset[$i]);
}
$gabung_dataset = Array();
foreach ($pertanyaan_dataset as $key => $value) {
	$gabung_dataset[$key]['pertanyaan'] = $pertanyaan_dataset[$key];
	$gabung_dataset[$key]['hasil_praproses'] = $hasilStemming_dataset[$key];
	$gabung_dataset[$key]['jawaban'] = $jawaban_dataset[$key];
	$gabung_dataset[$key]['label']= $labeldataset[$key];
}
print_r("<hr><h1>DATASET</h1>");
print_r($gabung_dataset);
print_r("</pre>");

// print_r('<pre>');
// print_r($hasilStem_dataset);

//SIMPAN DATASET KE DATABASE PERTANYAAN MYSQL
mysqli_query($connect, "TRUNCATE TABLE dataset");
foreach ($gabung_dataset as $row) {
	mysqli_query($connect, "INSERT INTO dataset ( id, pertanyaan, hasil_praproses, jawaban, label) VALUES ('','$row[pertanyaan]','$row[hasil_praproses]','$row[jawaban]','$row[label]')");
}
// ======================================================================================================================================== //

$dataset_baru = new Labeled($gabung_dataset, $labeldataset);
[$training, $testing] = $dataset_baru->randomize()->stratifiedSplit(0.8);

//AMBIL DATA LATIH
$datalatih = $training->getSamples();
$labeldatalatih = $training->getTargets();
//AMBIL DATA UJI
$datauji = $testing->getSamples();
$labeldatauji= $testing->getTargets();

// GABUNG DATA LATIH Dengan Label
print_r("<hr><pre><h1>DATA LATIH</h1><br>");
$gabung_datalatih = Array();
foreach ($datalatih as $kunci => $value) {
	$gabung_datalatih[$kunci]['pertanyaan'] = $value[0];
	$gabung_datalatih[$kunci]['hasil_praproses'] = $value[1];
	$gabung_datalatih[$kunci]['jawaban'] = $value[2];
	$gabung_datalatih[$kunci]['label'] = $labeldatalatih[$kunci];
}
print_r($gabung_datalatih);
print_r("</pre>");

print_r("<hr><pre><h1>DATA UJI</h1><br>");
$gabung_datauji = Array();
foreach ($datauji as $key => $value) {
	$gabung_datauji[$key]['pertanyaan'] = $value[0];
	$gabung_datauji[$key]['hasil_praproses'] = $value[1];
	$gabung_datauji[$key]['jawaban'] = $value[2];
	$gabung_datauji[$key]['label']= $labeldatauji[$key];
}
print_r($gabung_datauji);
print_r("</pre>");

//SIMPAN DATA LATIH KE DATABASE PERTANYAAN MYSQL
mysqli_query($connect, "TRUNCATE TABLE datalatih");
foreach ($gabung_datalatih as $row) {
	mysqli_query($connect, "INSERT INTO datalatih ( id, pertanyaan, hasil_praproses, jawaban, label) VALUES ('','$row[pertanyaan]','$row[hasil_praproses]','$row[jawaban]','$row[label]')");
}

//SIMPAN DATA UJI KE DATABASE UJI MYSQL
mysqli_query($connect, "TRUNCATE TABLE datauji");
foreach ($gabung_datauji as $row) {
	mysqli_query($connect, "INSERT INTO datauji ( id, pertanyaan, hasil_praproses, jawaban, label) VALUES ('','$row[pertanyaan]','$row[hasil_praproses]','$row[jawaban]','$row[label]')");
}

// ======================================================================================================================================== //


?>