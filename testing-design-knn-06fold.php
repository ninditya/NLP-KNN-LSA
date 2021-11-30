<?php
declare(strict_types=1);
namespace IFVABOTKNN;

require './vendor/autoload.php';
// ini_set('memory_limit', '1024M'); // or you could use 1G
ini_set('memory_limit', '-1');

use Rubix\ML\GridSearch;
use Rubix\ML\ModelManager;
use Rubix\ML\Backends\Tasks\TrainAndValidate;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Classifiers\SVC;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Kernels\SVM\Linear;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Tfidf;
use svmmodel;
use svm;

include './fungsi.php';
include './fungsi_svd.php';

$connect = connectDB('ifva');

//AMBIL DATA SET
$i=0;
$data_samples=mysqli_query($connect, "SELECT * FROM `dataset`");
$samples= array();
while ($d=mysqli_fetch_array($data_samples)) {
	$s['pertanyaan'] = $d['pertanyaan'];
	$s['hasil_praproses'] = $d['hasil_praproses'];
	$s['jawaban'] = $d['jawaban'];
	$samples[$i] = $s;
	$lab[$i] = $d['label'];
	$i++;
}
$kfold=6;
$dataset = new Labeled($samples, $lab);
// $folds = $dataset->randomize()->stratifiedFold($kfold);
$folds = $dataset->stratifiedFold($kfold);
// print_r("<pre>");
// print_r($folds);
// mysqli_query($connect, "TRUNCATE TABLE kfold_datauji_svm");
for ($i = 0; $i < $kfold; $i++) {
	$data[$i] = $folds[$i]->getSamples();
	$label[$i] = $folds[$i]->getTargets();
	foreach ($data[$i] as $key => $value) {
		$data[$i][$key][3] = $label[$i][$key];
		$pertanyaan[$i][$key] = $value[0];
		$hasil_praproses[$i][$key] = $value[1];
		$jawaban[$i][$key] = $value[2];
	}
	$x=0;
	// foreach ($data[$i] as $row) {
	// 	mysqli_query($connect, "INSERT INTO kfold_datauji_svm ( id, kfold, i, pertanyaan, hasil_praproses, jawaban, label) VALUES ('', $i, '$x', '$row[0]','$row[1]','$row[2]','$row[3]')");
	// 	$x++;
	// }
}
for ($i = 0; $i < $kfold; $i++) {
	$coba[$i][$i] = $i;
	$pertanyaandatalatih[$i] = array_diff_key($pertanyaan, $coba[$i]);
	$hasil_praprosesdatalatih[$i] = array_diff_key($hasil_praproses, $coba[$i]);
	$jawabandatalatih[$i] = array_diff_key($jawaban, $coba[$i]);
	$labeldatalatih[$i] = array_diff_key($label, $coba[$i]);
}
for ($i = 0; $i < $kfold; $i++) {
	$n=0;
	foreach ($pertanyaandatalatih[$i] as $key => $value) {
		foreach ($value as $kunci => $val) { 
			$pertanyaandatalatihbaru[$i][$n] = $pertanyaandatalatih[$i][$key][$kunci];
			$kalimatstemm[$i][$n] = $hasil_praprosesdatalatih[$i][$key][$kunci];
			$labeldatalatihbaru[$i][$n] = $labeldatalatih[$i][$key][$kunci];
			$jawabandatalatihbaru[$i][$n] = $jawabandatalatih[$i][$key][$kunci];
			$kalimatarraystemm[$i][$n] = explode(" ", $kalimatstemm[$i][$n]);
			$n++;
		}
	}
	$m=0;
	foreach($kalimatarraystemm[$i] as $key => $value ){
		foreach ($value as $kunci => $val){
			$termstemm[$i][$m] = $val;
			$m++;
		}
	}
	// print_r("</pre><hr><pre><h1>TFIDF DATALATIH MASING-MASING KFOLD KE- ".$i."</h1><br>");
	$con=new Tfidf();
	$con->proses($kalimatstemm[$i],$termstemm[$i]);
	$datalatih_table1[$i]=$con->table1;
	foreach ($datalatih_table1[$i] as $key => $value) {
		$datalatih_term[$i][$key] = array('term'=> $value['term']);
	}
	// TF DATA LATIH
	foreach ($kalimatarraystemm[$i] as $key => $value) {
		$tf_latih[$i][$key] = query($kalimatarraystemm[$i][$key], $datalatih_term[$i]);
	}
	//TF-IDF NORMALISASI L2
	$datalatih_table2[$i]=normalisasi($con->table2);
	$datalatih[$i] = new Labeled($datalatih_table2[$i], $labeldatalatihbaru[$i]);
	$array_latih[$i] = array('tf'=> $tf_latih[$i],'tfidf' => $datalatih_table2[$i], 'label' => $labeldatalatihbaru[$i]);
	//TF DATA UJI
	$x=0;
	foreach ($hasil_praproses[$i] as $key => $value) {
		$kalimatujiarraystemm[$i][$x] = explode(" ", $value);
		$tf_uji[$i][$x] = query($kalimatujiarraystemm[$i][$x], $datalatih_term[$i]);
		$array_uji[$i][$x] = array('tf' => $tf_uji[$i][$x] , 'label' => $label[$i][$x]);
		$x++;
	}
	$datauji[$i] = new Labeled($tf_uji[$i], $label[$i]);
}

// encode array to json array latih
$json_datalatih = json_encode($array_latih);
//write json to file
if (file_put_contents("json/datalatih-knn-6-fold.json", $json_datalatih))
	echo "<br>JSON DATA LATIH file created successfully...<br>";
else 
	echo "Oops! Error creating json file...";

// // encode array to json array uji
$json_datauji = json_encode($array_uji);
//write json to file
if (file_put_contents("json/datauji-knn-6-fold.json", $json_datauji))
	echo "<br>JSON DATA UJI file created successfully...<br>";
else 
	echo "Oops! Error creating json file...";

foreach ($pertanyaandatalatihbaru as $key => $value) {
	$i=0;
	foreach ($value as $kunci => $val) {
		$gabung_datalatih[$key][$i]['pertanyaan'] = $val;
		$gabung_datalatih[$key][$i]['hasil_praproses'] = $kalimatstemm[$key][$i];
		$gabung_datalatih[$key][$i]['jawaban'] = $jawabandatalatihbaru[$key][$i];
		$gabung_datalatih[$key][$i]['label'] = $labeldatalatihbaru[$key][$i];
		$i++;
	}
}
// mysqli_query($connect, "TRUNCATE TABLE kfold_datalatih_svm");
// for ($i = 0; $i < $kfold; $i++) {
// 	$x=0;
// 	foreach ($gabung_datalatih[$i] as $row) {
// 		mysqli_query($connect, "INSERT INTO kfold_datalatih_svm ( id, kfold, i, pertanyaan, hasil_praproses, jawaban, label) VALUES ('', $i, '$x', '$row[pertanyaan]','$row[hasil_praproses]','$row[jawaban]','$row[label]')");
// 		$x++;
// 	}
// }

//==============================================================================================================================================//

$report = new MulticlassBreakdown();
print_r('<pre><h3>PENGUJIAN K-FOLD KNearestNeighbors K = 5</h3>');
$estimator = new KNearestNeighbors(5);

for ($i = 0; $i < $kfold; $i++) {
	$estimator->train($datalatih[$i]);
	$predictions[$i] = $estimator->predict($datauji[$i]);
	$results[$i] = $report->generate($predictions[$i], $label[$i]);
	$akurasi[$i] = $results[$i]['overall']['accuracy'];
	$fmeasure[$i] = $results[$i]['overall']['f1_score'];
	$presisi[$i] = $results[$i]['overall']['precision'];
	$recall[$i] = $results[$i]['overall']['recall'];
	print_r("<br>AKURASI KE-   " . $i . " = " . $results[$i]['overall']['accuracy']);
	print_r("<br>F-MEASURE KE- " . $i . " = " . $results[$i]['overall']['f1_score']);
	print_r("<br>PRECISION KE- " . $i . " = " . $results[$i]['overall']['precision']);
	print_r("<br>RECALL KE-    " . $i . " = " . $results[$i]['overall']['recall']);
	print_r("<br>");
}
print_r("<br>RATA-RATA");
print_r("<br>AKURASI   = " . Stats::mean($akurasi));
print_r("<br>F-MEASURE = " . Stats::mean($fmeasure));
print_r("<br>PRESISI   = " . Stats::mean($presisi));
print_r("<br>RECALL    = " . Stats::mean($recall));
print_r("<br><br>");
print_r($results);

//==============================================================================================================================================//

// foreach ($predictions as $key => $value) {
// 	foreach($value as $kunci => $val){
// 		$data_kandidat[$key][$kunci] = mysqli_query($connect, "SELECT * FROM `kfold_datalatih` WHERE label = '$val' ");
// 		while ($d=mysqli_fetch_array($data_kandidat[$key][$kunci])) {
// 			$ans[$key][$kunci][$d['i']] = $d['jawaban'];
// 			$ask[$key][$kunci][$d['i']] = $d['pertanyaan'];
// 			foreach ($ask[$key][$kunci] as $i => $val) {
// 				$tfidf_kandidat[$key][$i] = array($datalatih_table2[$key][$i]);
// 				$nilaicossim[$key][$kunci][$i] = cosinesimilarity(array($tf_uji[$key][$kunci]), matrixTranspose($tfidf_kandidat[$key][$i]));
// 				$k[$key][$kunci][$i] = $nilaicossim[$key][$kunci][$i][0][0]; 
// 				arsort($k[$key][$kunci], SORT_NUMERIC);
// 			}
// 		}
// 		$x=0;
// 		foreach ($k[$key][$kunci] as $y => $value) {
// 			$kandidat[$key][$kunci][$x]['nilaicossim'] = $value;
// 			$kandidat[$key][$kunci][$x]['pertanyaan'] = $ask[$key][$kunci][$y];
// 			$kandidat[$key][$kunci][$x]['jawaban'] = $ans[$key][$kunci][$y];
// 			$x++;
// 		}
// 	}
// }
// // print_r($predictions);
// // print_r($ans);
// foreach ($kandidat as $key => $value) {
// 	foreach ($value as $k => $val) {
// 		$hasilakhir[$key][$k] = $val[0]['jawaban'];
// 	}
// }
// // print_r($hasilakhir);
// for ($i = 0; $i < $kfold; $i++) {
// 	$resultsretrieval[$i] = $report->generate($hasilakhir[$i], $jawaban[$i]);
// 	$akurasiretrieval[$i] = $resultsretrieval[$i]['overall']['accuracy'];
// 	$fmeasureretrieval[$i] = $resultsretrieval[$i]['overall']['f1_score'];
// 	$presisiretrieval[$i] = $resultsretrieval[$i]['overall']['precision'];
// 	$recallretrieval[$i] = $resultsretrieval[$i]['overall']['recall'];
// 	print_r("<br>AKURASI KE-   " . $i . " = " . $resultsretrieval[$i]['overall']['accuracy']);
// 	print_r("<br>F-MEASURE KE- " . $i . " = " . $resultsretrieval[$i]['overall']['f1_score']);
// 	print_r("<br>PRECISION KE- " . $i . " = " . $resultsretrieval[$i]['overall']['precision']);
// 	print_r("<br>RECALL KE-    " . $i . " = " . $resultsretrieval[$i]['overall']['recall']);
// 	print_r("<br>");
// }
// // print_r($resultsretrieval);
// print_r("<br>RATA-RATA");
// print_r("<br>AKURASI   = " . Stats::mean($akurasiretrieval));
// print_r("<br>F-MEASURE = " . Stats::mean($fmeasureretrieval));
// print_r("<br>PRESISI   = " . Stats::mean($presisiretrieval));
// print_r("<br>RECALL    = " . Stats::mean($recallretrieval));
// print_r("<br><br>");

// //===========================================================================================================================================//

// print_r($predictions[0][0]);

// arsort($akurasi, SORT_NUMERIC);
// $id_akurasiurut=array_keys($akurasi);
// print_r($id_akurasiurut);

// if($akurasi[$id_akurasiurut[0]] == $akurasi[$id_akurasiurut[1]]){
// 	print_r("<br>");
// 	if ($fmeasure[$id_akurasiurut[0]] >= $fmeasure[$id_akurasiurut[1]]) {
// 		print_r("id paling bagus = " . $id_akurasiurut[0] . "<br><br>");
// 		$datalatihbagus = $datalatihpengujian[$id_akurasiurut[0]];
// 		$dataujibagus = $folds[$id_akurasiurut[0]];
// 	} else {
// 		print_r("id paling bagus = " . $id_akurasiurut[1] . "<br><br>");
// 		$datalatihbagus = $datalatihpengujian[$id_akurasiurut[1]];
// 		$dataujibagus = $folds[$id_akurasiurut[1]];
// 	}
// } else {
// 	print_r("<br>");
// 	print_r("id paling bagus = " . $id_akurasiurut[0] . "<br><br>");
// 	$datalatihbagus = $datalatihpengujian[$id_akurasiurut[0]];
// 	$dataujibagus = $folds[$id_akurasiurut[0]];
// 	$semuadata = $semuadatalatih[$id_akurasiurut[0]];
// }

//==============================================================================================================================================//

?>