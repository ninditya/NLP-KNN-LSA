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

$data_uji = file_get_contents("json/datauji-knn-9-fold.json");
$json_uji = json_decode($data_uji, TRUE);
print_r("<pre>");
foreach ($json_uji[3] as $key => $value) {
	$labeluji[3][$key] = $value['label'];
}
// print_r($labeluji);

$data_latih = file_get_contents("json/datalatih-knn-9-fold.json");
$json_latih = json_decode($data_latih, TRUE);
// print_r("<pre>");
// print_r($json_latih[3]['tfidf']);
// print_r("<br>");

// print_r("<hr><h1>SVD datalatih</h1><br>");
$d_datalatih[3] = count($json_latih[3]['tfidf']);
$t_datalatih[3] = count(matrixTranspose($json_latih[3]['tfidf']));
if ($t_datalatih[3] > $d_datalatih[3]) {
	$matrix_datalatih[3] = matrixTranspose($json_latih[3]['tfidf']);
} elseif ( $d_datalatih[3] >= $t_datalatih[3]) {
	$matrix_datalatih[3] = $json_latih[3]['tfidf'];
}
print_r("TERM DATA LATIH = ".$t_datalatih[3]);
print_r("<br>");
print_r("DOKUMEN DATA LATIH = ".$d_datalatih[3]);

if (!empty($matrix_datalatih[3])) {
	$svd_datalatih[3] = SVD($matrix_datalatih[3]);
}

// $ini_k[0] = intval(428);
// $ini_k[1] = intval(430);
// $ini_k[2] = intval(432);
// $ini_k[3] = intval(434);
// $ini_k[4] = intval(438);
// $ini_k[5] = intval(440);
// $ini_k[6] = intval(442);
// $ini_k[0] = intval($svd_datalatih[3]['K']*5/100);
// $ini_k[1] = intval($svd_datalatih[3]['K']*10/100);
// $ini_k[2] = intval($svd_datalatih[3]['K']*15/100);
// $ini_k[3] = intval($svd_datalatih[3]['K']*20/100);
// $ini_k[4] = intval($svd_datalatih[3]['K']*25/100);
// $ini_k[5] = intval($svd_datalatih[3]['K']*30/100);
// $ini_k[6] = intval($svd_datalatih[3]['K']*35/100);
// $ini_k[7] = intval($svd_datalatih[3]['K']*40/100);
// $ini_k[8] = intval($svd_datalatih[3]['K']*45/100);
// $ini_k[9] = intval($svd_datalatih[3]['K']*50/100);
// $ini_k[10] = intval($svd_datalatih[3]['K']*55/100);
// $ini_k[11] = intval($svd_datalatih[3]['K']*60/100);
// $ini_k[12] = intval($svd_datalatih[3]['K']*65/100);
// $ini_k[13] = intval($svd_datalatih[3]['K']*70/100);
// $ini_k[14] = intval($svd_datalatih[3]['K']*75/100);
// $ini_k[15] = intval($svd_datalatih[3]['K']*80/100);
// $ini_k[16] = intval($svd_datalatih[3]['K']*85/100);
// $ini_k[17] = intval($svd_datalatih[3]['K']*90/100);
// $ini_k[18] = intval($svd_datalatih[3]['K']*95/100);
// $ini_k[19] = intval($svd_datalatih[3]['K']);

foreach ($ini_k as $key => $value) {
	$svd_datalatih_red[3][$key] = array();
	$svd_datalatih_red[3][$key]['sp'] = matrixConstruct($svd_datalatih[3]['S'], $value, $value);
	$svd_datalatih_red[3][$key]['up'] = matrixConstruct($svd_datalatih[3]['U'], count($svd_datalatih[3]['U']), $value);
	$svd_datalatih_red[3][$key]['vtp'] = matrixConstruct(matrixTranspose($svd_datalatih[3]['V']), $value, count($svd_datalatih[3]['V']));
	$newArr[3][$key] = Array();
	foreach ($svd_datalatih_red[3][$key]['sp']  as $k => $val) {
		$r = Array();
		foreach ($val as $v) {
			array_push($r, floatval($v) == 0.0 ? 0 : floatval($v));
		}
		array_push($newArr[3][$key], $r);
	}
	$spinvers_datalatih[3][$key] = invert($newArr[3][$key]);
	// echo "<hr><pre><h1>LATENT SEMANTIC ANALYSIS LATIH</h1>";
	if ($t_datalatih[3] > $d_datalatih[3]) {
		// print_r("<br> disini buat t > d rumusnya q=qTUkS-1");
		// $red[$key] = $svd_datalatih_red[3][$key]['up'];
		$red[$key] = perkalian_matrix($svd_datalatih_red[3][$key]['up'], $spinvers_datalatih[3][$key]);
	} elseif ( $d_datalatih[3] >= $t_datalatih[3]) {
		// print_r("<br>disini buat d > t rumusnya q=qTVTkS-1");
		// $red[$key] = matrixTranspose($svd_datalatih_red[3][$key]['vtp']);
		$red[$key] = perkalian_matrix(matrixTranspose($svd_datalatih_red[3][$key]['vtp']), $spinvers_datalatih[3][$key]);
	}
	// REDUKSI PADA DATA LATIH
	for ($x=0; $x < $d_datalatih[3] ; $x++) {
		$tf_datalatih_red[3][$key][$x] = perkalian_matrix(array($json_latih[3]['tfidf'][$x]), $red[$key]);
		foreach ($tf_datalatih_red[3][$key] as $k => $value) {
			foreach ($value as $kunci => $val) {
				$svd_datalatih[3][$key]['red'][$k] = $val;
			}
		}
	}
	//REDUKSI PADA DATA UJI
	$x=0;
	foreach ($json_uji[3] as $k => $value) {
		$tf_uji_red[3][$key][$k] = perkalian_matrix(array($json_uji[3][$k]['tf']),$red[$key]);
		foreach ($tf_uji_red[3][$key] as $ke => $value) {
			foreach ($value as $kunci => $val) {
				$svd_datauji[3][$key][$ke] = $val;
			}
		}
	}
	$datalatihred[3][$key] = new Labeled($svd_datalatih[3][$key]['red'], $json_latih[3]['label']);
	$dataujired[3][$key] = new Labeled($svd_datauji[3][$key], $labeluji[3]);

// //===========================================================================================================================================//

	$report = new MulticlassBreakdown();
	print_r('<pre><h3>PENGUJIAN FOLD ke 0 KNearestNeighbors K = 5 LSA K ='.$ini_k[$key].'</h3>');
	$estimator = new KNearestNeighbors(5);
	
	$estimator->train($datalatihred[3][$key]);
	$predictions[3][$key] = $estimator->predict($dataujired[3][$key]);
	$results[3][$key] = $report->generate($predictions[3][$key], $labeluji[3]);
	$akurasi[3][$key] = $results[3][$key]['overall']['accuracy'];
	$fmeasure[3][$key] = $results[3][$key]['overall']['f1_score'];
	$presisi[3][$key] = $results[3][$key]['overall']['precision'];
	$recall[3][$key] = $results[3][$key]['overall']['recall'];
	print_r("<br>AKURASI KE-   " . 3 . " = " . $results[3][$key]['overall']['accuracy']);
	print_r("<br>F-MEASURE KE- " . 3 . " = " . $results[3][$key]['overall']['f1_score']);
	print_r("<br>PRECISION KE- " . 3 . " = " . $results[3][$key]['overall']['precision']);
	print_r("<br>RECALL KE-    " . 3 . " = " . $results[3][$key]['overall']['recall']);
	print_r("<br><br>");
	// print_r($results[3][$key]);

//===========================================================================================================================================//

}


// $svd_datalatih_red[0] = array();
// $svd_datalatih_red[0]['sp'] = matrixConstruct($svd_datalatih[0]['S'], $ini_k[0], $ini_k[0]);
// $svd_datalatih_red[0]['up'] = matrixConstruct($svd_datalatih[0]['U'], count($svd_datalatih[0]['U']), $ini_k[0]);
// $svd_datalatih_red[0]['vtp'] = matrixConstruct(matrixTranspose($svd_datalatih[0]['V']), $ini_k[0], count($svd_datalatih[0]['V']));
// $newArr[0] = Array();
// foreach ($svd_datalatih_red[0]['sp']  as $key => $val) {
// 	$r = Array();
// 	foreach ($val as $v) {
// 		array_push($r, floatval($v) == 0.0 ? 0 : floatval($v));
// 	}
// 	array_push($newArr[0], $r);
// }
// $spinvers_datalatih[0] = invert($newArr[0]);

// // echo "<hr><pre><h1>LATENT SEMANTIC ANALYSIS LATIH</h1>";
// if ($t_datalatih[0] > $d_datalatih[0]) {
// 	// print_r("<br> disini buat t > d rumusnya q=qTUkS-1");
// 	//REDUKSI PADA DATA LATIH DENGAN STEMMING
// 	for ($x=0; $x < $d_datalatih[0] ; $x++) {
// 		$tf_datalatih_r[0][$x] = perkalian_matrix(array($json_latih[0]['tf'][$x]), $svd_datalatih_red[0]['up']);
// 		$tf_datalatih_red[0][$x] = perkalian_matrix($tf_datalatih_r[0][$x],$spinvers_datalatih[0]);
// 		foreach ($tf_datalatih_red[0] as $key => $value) {
// 			foreach ($value as $kunci => $val) {
// 				$svd_datalatih[0]['red'][$key] = $val;
// 			}
// 		}
// 	}
// 	// $datalatihred[0] = new Labeled($svd_datalatih[0]['red'], $json_latih[0]['label']);
// 	//REDUKSI PADA DATA UJI
// 	foreach ($json_uji[0] as $key => $value) {
// 		$tf_uji_r[0][$key] = perkalian_matrix(array($json_uji[0][$key]['tf']), $svd_datalatih_red[0]['up']);
// 		$tf_uji_red[0][$key] = perkalian_matrix($tf_uji_r[0][$key],$spinvers_datalatih[0]);
// 		foreach ($tf_uji_red[0] as $k => $value) {
// 			foreach ($value as $kunci => $val) {
// 				$svd_datauji[0][$k] = $val;
// 			}
// 		}
// 	}
// } elseif ( $d_datalatih[0] >= $t_datalatih[0]) {
// 		// print_r("<br>disini buat d > t rumusnya q=qTVTkS-1");
// 		//REDUKSI PADA DATA LATIH DENGAN STEMMING
// 	for ($x=0; $x < $d_datalatih[0] ; $x++) {
// 		$tf_datalatih_r[0][$x] = perkalian_matrix(array($json_latih[0]['tf'][$x]), matrixTranspose($svd_datalatih_red[0]['vtp']));
// 		$tf_datalatih_red[0][$x] = perkalian_matrix($tf_datalatih_r[0][$x],$spinvers_datalatih[0]);
// 		foreach ($tf_datalatih_red[0] as $key => $value) {
// 			foreach ($value as $kunci => $val) {
// 				$svd_datalatih[0]['red'][$key] = $val;
// 			}
// 		}
// 	}
// 	//REDUKSI PADA DATA UJI
// 	$x=0;
// 	foreach ($json_uji[0] as $key => $value) {
// 		$tf_uji_r[0][$key] = perkalian_matrix(array($json_uji[0][$key]['tf']), matrixTranspose($svd_datalatih_red[0]['vtp']));
// 		$tf_uji_red[0][$key] = perkalian_matrix($tf_uji_r[0][$key],$spinvers_datalatih[0]);
// 		foreach ($tf_uji_red[0] as $k => $value) {
// 			foreach ($value as $kunci => $val) {
// 				$svd_datauji[0][$k] = $val;
// 			}
// 		}
// 	}
// }
// $datalatihred[0] = new Labeled($svd_datalatih[0]['red'], $json_latih[0]['label']);
// $dataujired[0] = new Labeled($svd_datauji[0], $labeluji[0]);

// //===========================================================================================================================================//

// $report = new MulticlassBreakdown();
// print_r('<pre><h3>PENGUJIAN K-FOLD KNearestNeighbors K = 5</h3>');
// $estimator = new SVC(1.0, new Linear(), true, 1e-5, 50.0);

// $estimator->train($datalatihred[0]);
// $predictions[0] = $estimator->predict($dataujired[0]);
// $results[0] = $report->generate($predictions[0], $labeluji[0]);
// $akurasi[0] = $results[0]['overall']['accuracy'];
// $fmeasure[0] = $results[0]['overall']['f1_score'];
// $presisi[0] = $results[0]['overall']['precision'];
// $recall[0] = $results[0]['overall']['recall'];
// print_r("<br>AKURASI KE-   " . 0 . " = " . $results[0]['overall']['accuracy']);
// print_r("<br>F-MEASURE KE- " . 0 . " = " . $results[0]['overall']['f1_score']);
// print_r("<br>PRECISION KE- " . 0 . " = " . $results[0]['overall']['precision']);
// print_r("<br>RECALL KE-    " . 0 . " = " . $results[0]['overall']['recall']);
// print_r("<br><br>");

// print_r($results);
// print_r("<br>RATA-RATA");
// print_r("<br>AKURASI   = " . Stats::mean($akurasi));
// print_r("<br>F-MEASURE = " . Stats::mean($fmeasure));
// print_r("<br>PRESISI   = " . Stats::mean($presisi));
// print_r("<br>RECALL    = " . Stats::mean($recall));
// print_r("<br><br>");

//===========================================================================================================================================//



// 	// echo "<hr><pre><h1>LATENT SEMANTIC ANALYSIS LATIH</h1>";
// 	if ($t_datalatih[$i] > $d_datalatih[$i]) {
// 		// print_r("<br> disini buat t > d rumusnya q=qTUkS-1");
// 		//REDUKSI PADA DATA LATIH DENGAN STEMMING
// 		for ($x=0; $x < $d_datalatih[$i] ; $x++) {
// 			$tf_datalatih_r[$i][$x] = perkalian_matrix($tf_datalatih[$i][$x], $svd_datalatih_red[$i]['up']);
// 			$tf_datalatih_red[$i][$x] = perkalian_matrix($tf_datalatih_r[$i][$x],$spinvers_datalatih[$i]);

// 			foreach ($tf_datalatih_red[$i] as $key => $value) {
// 				foreach ($value as $kunci => $val) {
// 					$svd_datalatih[$i]['red'][$key] = $val;
// 				}
// 			}
// 		}
// 		$datalatihred[$i] = new Labeled($svd_datalatih[$i]['red'], $labeldatalatihbaru[$i]);
// 		//REDUKSI PADA DATA UJI
// 		$x=0;
// 		foreach ($hasil_praproses[$i] as $key => $value) {
// 			$kalimatujiarraystemm[$i][$x] = explode(" ", $value);
// 			$tf_uji[$i][$x][0] = query($kalimatujiarraystemm[$i][$x], $datalatih_term[$i]);
// 			$tf_uji_r[$i][$x] = perkalian_matrix($tf_uji[$i][$x], $svd_datalatih_red[$i]['up']);
// 			$tf_uji_red[$i][$x] = perkalian_matrix($tf_uji_r[$i][$x],$spinvers_datalatih[$i]);
// 			$x++;

// 			foreach ($tf_uji_red[$i] as $key => $value) {
// 				foreach ($value as $kunci => $val) {
// 					$svd_datauji[$i][$key] = $val;
// 				}
// 			}
// 		}
// 		$dataujired[$i] = new Labeled($svd_datauji[$i], $label[$i]);

// 	} elseif ( $d_datalatih[$i] >= $t_datalatih[$i]) {
// 		// print_r("<br>disini buat d > t rumusnya q=qTVTkS-1");
// 		//REDUKSI PADA DATA LATIH DENGAN STEMMING
// 		for ($x=0; $x < $d_datalatih[$i] ; $x++) {
// 			$tf_datalatih[$i][$x][0] = query($kalimatarraystemm[$i][$x], $datalatih_term[$i]);
// 			$tf_datalatih_r[$i][$x] = perkalian_matrix($tf_datalatih[$i][$x], matrixTranspose($svd_datalatih_red[$i]['vtp']));
// 			$tf_datalatih_red[$i][$x] = perkalian_matrix($tf_datalatih_r[$i][$x],$spinvers_datalatih[$i]);

// 			foreach ($tf_datalatih_red[$i] as $key => $value) {
// 				foreach ($value as $kunci => $val) {
// 					$svd_datalatih[$i]['red'][$key] = $val;
// 				}
// 			}
// 		}
// 		$datalatihred[$i] = new Labeled($svd_datalatih[$i]['red'], $labeldatalatihbaru[$i]);
// 		//REDUKSI PADA DATA UJI
// 		$x=0;
// 		foreach ($hasil_praproses[$i] as $key => $value) {
// 			$kalimatujiarraystemm[$i][$x] = explode(" ", $value);
// 			$tf_uji[$i][$x][0] = query($kalimatujiarraystemm[$i][$x], $datalatih_term[$i]);
// 			$tf_uji_r[$i][$x] = perkalian_matrix($tf_uji[$i][$x], matrixTranspose($svd_datalatih_red[$i]['vtp']));
// 			$tf_uji_red[$i][$x] = perkalian_matrix($tf_uji_r[$i][$x],$spinvers_datalatih[$i]);
// 			$x++;

// 			foreach ($tf_uji_red[$i] as $key => $value) {
// 				foreach ($value as $kunci => $val) {
// 					$svd_datauji[$i][$key] = $val;
// 				}
// 			}
// 		}
// 		$dataujired[$i] = new Labeled($svd_datauji[$i], $label[$i]);
// 	}
// }



// //AMBIL DATA SET
// $i=0;
// $data_samples=mysqli_query($connect, "SELECT * FROM `dataset`");
// $samples= array();
// while ($d=mysqli_fetch_array($data_samples)) {
// 	$s['pertanyaan'] = $d['pertanyaan'];
// 	$s['hasil_praproses'] = $d['hasil_praproses'];
// 	$s['jawaban'] = $d['jawaban'];
// 	$samples[$i] = $s;
// 	$lab[$i] = $d['label'];
// 	$i++;
// }
// $kfold=5;
// $dataset = new Labeled($samples, $lab);
// $folds = $dataset->randomize()->stratifiedFold($kfold);
// // $folds = $dataset->stratifiedFold($kfold);
// mysqli_query($connect, "TRUNCATE TABLE kfold_datauji_lsa");
// for ($i = 0; $i < $kfold; $i++) {
// 	$data[$i] = $folds[$i]->getSamples();
// 	$label[$i] = $folds[$i]->getTargets();
// 	foreach ($data[$i] as $key => $value) {
// 		$data[$i][$key][3] = $label[$i][$key];
// 		$pertanyaan[$i][$key] = $value[0];
// 		$hasil_praproses[$i][$key] = $value[3];
// 		$jawaban[$i][$key] = $value[2];
// 	}
// 	$x=0;
// 	foreach ($data[$i] as $row) {
// 		mysqli_query($connect, "INSERT INTO kfold_datauji_lsa ( id, kfold, i, pertanyaan, hasil_praproses, jawaban, label) VALUES ('', $key, '$x', '$row[0]','$row[1]','$row[2]','$row[3]')");
// 		$x++;
// 	}
// }
// for ($i = 0; $i < $kfold; $i++) {
// 	$coba[$i][$i] = $i;
// 	$pertanyaandatalatih[$i] = array_diff_key($pertanyaan, $coba[$i]);
// 	$hasil_praprosesdatalatih[$i] = array_diff_key($hasil_praproses, $coba[$i]);
// 	$jawabandatalatih[$i] = array_diff_key($jawaban, $coba[$i]);
// 	$labeldatalatih[$i] = array_diff_key($label, $coba[$i]);
// }
// for ($i = 0; $i < $kfold; $i++) {
// 	$n=0;
// 	foreach ($pertanyaandatalatih[$i] as $key => $value) {
// 		foreach ($value as $kunci => $val) { 
// 			$pertanyaandatalatihbaru[$i][$n] = $pertanyaandatalatih[$i][$key][$kunci];
// 			$kalimatstemm[$i][$n] = $hasil_praprosesdatalatih[$i][$key][$kunci];
// 			$labeldatalatihbaru[$i][$n] = $labeldatalatih[$i][$key][$kunci];
// 			$jawabandatalatihbaru[$i][$n] = $jawabandatalatih[$i][$key][$kunci];
// 			$kalimatarraystemm[$i][$n] = explode(" ", $kalimatstemm[$i][$n]);
// 			$n++;
// 		}
// 	}
// 	$m=0;
// 	foreach($kalimatarraystemm[$i] as $key => $value ){
// 		foreach ($value as $kunci => $val){
// 			$termstemm[$i][$m] = $val;
// 			$m++;
// 		}
// 	}
// 	// print_r("</pre><hr><pre><h1>TFIDF DATALATIH MASING-MASING KFOLD KE- ".$i."</h1><br>");
// 	$con=new Tfidf();
// 	$con->proses($kalimatstemm[$i],$termstemm[$i]);
// 	$datalatih_table1[$i]=$con->table1;
// 	foreach ($datalatih_table1[$i] as $key => $value) {
// 		$datalatih_term[$i][$key] = array('term'=> $value['term']);
// 	}
// 	//TF-IDF NORMALISASI L2
// 	$datalatih_table2[$i]=normalisasi($con->table2);
// 	$d_datalatih[$i] = count($datalatih_table2[$i]);
// 	$t_datalatih[$i] = count(matrixTranspose($datalatih_table2[$i]));

// 	// print_r("<hr><h1>SVD datalatih</h1><br>");
// 	if ($t_datalatih[$i] > $d_datalatih[$i]) {
// 		$matrix_datalatih[$i] = matrixTranspose($datalatih_table2[$i]);
// 	} elseif ( $d_datalatih[$i] >= $t_datalatih[$i]) {
// 		$matrix_datalatih[$i] = $datalatih_table2[$i];
// 	}
// 	if (!empty($matrix_datalatih[$i])) {
// 		$svd_datalatih[$i] = SVD($matrix_datalatih[$i]);
// 	}

// 	$ini_k[$i] = $svd_datalatih[$i]['K']*75/100;
// 	$svd_datalatih_red[$i] = array();
// 	$svd_datalatih_red[$i]['sp'] = matrixConstruct($svd_datalatih[$i]['S'], $ini_k[$i], $ini_k[$i]);
// 	$svd_datalatih_red[$i]['up'] = matrixConstruct($svd_datalatih[$i]['U'], count($svd_datalatih[$i]['U']), $ini_k[$i]);
// 	$svd_datalatih_red[$i]['vtp'] = matrixConstruct(matrixTranspose($svd_datalatih[$i]['V']), $ini_k[$i], count($svd_datalatih[$i]['V']));

// 	$newArr[$i] = Array();
// 	foreach ($svd_datalatih_red[$i]['sp']  as $key => $val) {
// 		$r = Array();
// 		foreach ($val as $v) {
// 			array_push($r, floatval($v) == 0.0 ? 0 : floatval($v));
// 		}
// 		array_push($newArr[$i], $r);
// 	}
// 	$spinvers_datalatih[$i] = invert($newArr[$i]);

// 	// echo "<hr><pre><h1>LATENT SEMANTIC ANALYSIS LATIH</h1>";
// 	if ($t_datalatih[$i] > $d_datalatih[$i]) {
// 		// print_r("<br> disini buat t > d rumusnya q=qTUkS-1");
// 		//REDUKSI PADA DATA LATIH DENGAN STEMMING
// 		for ($x=0; $x < $d_datalatih[$i] ; $x++) {
// 			$tf_datalatih[$i][$x][0] = query($kalimatarraystemm[$i][$x], $datalatih_term[$i]);
// 			$tf_datalatih_r[$i][$x] = perkalian_matrix($tf_datalatih[$i][$x], $svd_datalatih_red[$i]['up']);
// 			$tf_datalatih_red[$i][$x] = perkalian_matrix($tf_datalatih_r[$i][$x],$spinvers_datalatih[$i]);

// 			foreach ($tf_datalatih_red[$i] as $key => $value) {
// 				foreach ($value as $kunci => $val) {
// 					$svd_datalatih[$i]['red'][$key] = $val;
// 				}
// 			}
// 		}
// 		$datalatihred[$i] = new Labeled($svd_datalatih[$i]['red'], $labeldatalatihbaru[$i]);
// 		//REDUKSI PADA DATA UJI
// 		$x=0;
// 		foreach ($hasil_praproses[$i] as $key => $value) {
// 			$kalimatujiarraystemm[$i][$x] = explode(" ", $value);
// 			$tf_uji[$i][$x][0] = query($kalimatujiarraystemm[$i][$x], $datalatih_term[$i]);
// 			$tf_uji_r[$i][$x] = perkalian_matrix($tf_uji[$i][$x], $svd_datalatih_red[$i]['up']);
// 			$tf_uji_red[$i][$x] = perkalian_matrix($tf_uji_r[$i][$x],$spinvers_datalatih[$i]);
// 			$x++;

// 			foreach ($tf_uji_red[$i] as $key => $value) {
// 				foreach ($value as $kunci => $val) {
// 					$svd_datauji[$i][$key] = $val;
// 				}
// 			}
// 		}
// 		$dataujired[$i] = new Labeled($svd_datauji[$i], $label[$i]);

// 	} elseif ( $d_datalatih[$i] >= $t_datalatih[$i]) {
// 		// print_r("<br>disini buat d > t rumusnya q=qTVTkS-1");
// 		//REDUKSI PADA DATA LATIH DENGAN STEMMING
// 		for ($x=0; $x < $d_datalatih[$i] ; $x++) {
// 			$tf_datalatih[$i][$x][0] = query($kalimatarraystemm[$i][$x], $datalatih_term[$i]);
// 			$tf_datalatih_r[$i][$x] = perkalian_matrix($tf_datalatih[$i][$x], matrixTranspose($svd_datalatih_red[$i]['vtp']));
// 			$tf_datalatih_red[$i][$x] = perkalian_matrix($tf_datalatih_r[$i][$x],$spinvers_datalatih[$i]);

// 			foreach ($tf_datalatih_red[$i] as $key => $value) {
// 				foreach ($value as $kunci => $val) {
// 					$svd_datalatih[$i]['red'][$key] = $val;
// 				}
// 			}
// 		}
// 		$datalatihred[$i] = new Labeled($svd_datalatih[$i]['red'], $labeldatalatihbaru[$i]);
// 		//REDUKSI PADA DATA UJI
// 		$x=0;
// 		foreach ($hasil_praproses[$i] as $key => $value) {
// 			$kalimatujiarraystemm[$i][$x] = explode(" ", $value);
// 			$tf_uji[$i][$x][0] = query($kalimatujiarraystemm[$i][$x], $datalatih_term[$i]);
// 			$tf_uji_r[$i][$x] = perkalian_matrix($tf_uji[$i][$x], matrixTranspose($svd_datalatih_red[$i]['vtp']));
// 			$tf_uji_red[$i][$x] = perkalian_matrix($tf_uji_r[$i][$x],$spinvers_datalatih[$i]);
// 			$x++;

// 			foreach ($tf_uji_red[$i] as $key => $value) {
// 				foreach ($value as $kunci => $val) {
// 					$svd_datauji[$i][$key] = $val;
// 				}
// 			}
// 		}
// 		$dataujired[$i] = new Labeled($svd_datauji[$i], $label[$i]);
// 	}
// }

// mysqli_query($connect, "TRUNCATE TABLE kfold_datalatih_lsa");
// foreach ($pertanyaandatalatihbaru as $key => $value) {
// 	$i=0;
// 	foreach ($value as $kunci => $val) {
// 		$gabung_datalatih[$key][$i]['pertanyaan'] = $val;
// 		$gabung_datalatih[$key][$i]['hasil_praproses'] = $kalimatstemm[$key][$i];
// 		$gabung_datalatih[$key][$i]['jawaban'] = $jawabandatalatihbaru[$key][$i];
// 		$gabung_datalatih[$key][$i]['label'] = $labeldatalatihbaru[$key][$i];
// 		$i++;
// 	}
// 	$x=0;
// 	foreach ($gabung_datalatih[$key] as $row) {
// 		mysqli_query($connect, "INSERT INTO kfold_datalatih_lsa ( id, kfold, i, pertanyaan, hasil_praproses, jawaban, label) VALUES ('', $key, '$x', '$row[pertanyaan]','$row[hasil_praproses]','$row[jawaban]','$row[label]')");
// 		$x++;
// 	}
// }
// //===========================================================================================================================================//

// $report = new MulticlassBreakdown();
// print_r('<pre><h3>PENGUJIAN K-FOLD KNearestNeighbors K = 5</h3>');
// $estimator = new SVC(1.0, new Linear(), true, 1e-5, 50.0);

// for ($i = 0; $i < $kfold; $i++) {
// 	$estimator->train($datalatihred[$i]);
// 	$predictions[$i] = $estimator->predict($dataujired[$i]);
// 	$results[$i] = $report->generate($predictions[$i], $label[$i]);
// 	$akurasi[$i] = $results[$i]['overall']['accuracy'];
// 	$fmeasure[$i] = $results[$i]['overall']['f1_score'];
// 	$presisi[$i] = $results[$i]['overall']['precision'];
// 	$recall[$i] = $results[$i]['overall']['recall'];
// 	print_r("<br>AKURASI KE-   " . $i . " = " . $results[$i]['overall']['accuracy']);
// 	print_r("<br>F-MEASURE KE- " . $i . " = " . $results[$i]['overall']['f1_score']);
// 	print_r("<br>PRECISION KE- " . $i . " = " . $results[$i]['overall']['precision']);
// 	print_r("<br>RECALL KE-    " . $i . " = " . $results[$i]['overall']['recall']);
// 	print_r("<br>");
// }
// print_r("<br>RATA-RATA");
// print_r("<br>AKURASI   = " . Stats::mean($akurasi));
// print_r("<br>F-MEASURE = " . Stats::mean($fmeasure));
// print_r("<br>PRESISI   = " . Stats::mean($presisi));
// print_r("<br>RECALL    = " . Stats::mean($recall));
// print_r("<br><br>");

// //===========================================================================================================================================//

// foreach ($predictions as $key => $value) {
// 	foreach($value as $kunci => $val){
// 		$data_kandidat[$key][$kunci] = mysqli_query($connect, "SELECT * FROM `kfold_datalatih_lsa` WHERE label = '$val' ");
// 		while ($d=mysqli_fetch_array($data_kandidat[$key][$kunci])) {
// 			$ans[$key][$kunci][$d['i']] = $d['jawaban'];
// 			$ask[$key][$kunci][$d['i']] = $d['pertanyaan'];
// 			foreach ($ask[$key][$kunci] as $i => $val) {
// 				$lsa_kandidat[$key][$i] = array($svd_datalatih[$key]['red'][$i]);
// 				$nilaicossim[$key][$kunci][$i] = cosinesimilaritylsa($tf_uji_red[$key][$kunci], matrixTranspose($lsa_kandidat[$key][$i]));
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
// foreach ($kandidat as $key => $value) {
// 	foreach ($value as $k => $val) {
// 		$hasilakhir[$key][$k] = $val[0]['jawaban'];
// 	}
// }

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
// print_r("<br>RATA-RATA");
// print_r("<br>AKURASI   = " . Stats::mean($akurasiretrieval));
// print_r("<br>F-MEASURE = " . Stats::mean($fmeasureretrieval));
// print_r("<br>PRESISI   = " . Stats::mean($presisiretrieval));
// print_r("<br>RECALL    = " . Stats::mean($recallretrieval));
// print_r("<br><br>");
// print_r($kalimatujiarraystemm[0][0]);
// print_r($datalatih_term[0]);

// //===========================================================================================================================================//

// declare(strict_types=1);
// namespace IFVABOTKNN;

// require './vendor/autoload.php';
// // ini_set('memory_limit', '1024M'); // or you could use 1G
// ini_set('memory_limit', '-1');

// use Rubix\ML\GridSearch;
// use Rubix\ML\ModelManager;
// use Rubix\ML\Classifiers\KNearestNeighbors;
// use Rubix\ML\Kernels\Distance\Manhattan;
// use Rubix\ML\Backends\Tasks\TrainAndValidate;
// use Rubix\ML\Classifiers\SVC;
// use Rubix\ML\CrossValidation\KFold;
// use Rubix\ML\CrossValidation\Metrics\Accuracy;
// use Rubix\ML\CrossValidation\Metrics\FBeta;
// use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
// use Rubix\ML\Datasets\Labeled;
// use Rubix\ML\Datasets\Dataset;
// use Rubix\ML\Persisters\Filesystem;
// use Rubix\ML\Kernels\SVM\Linear;
// use Rubix\ML\Other\Helpers\Stats;
// use svmmodel;
// use svm;

// include './fungsi.php';
// include './fungsi_svd.php';

// $connect = connectDB('ifva');

// //TABLE TFIDF
// $table2 = array();
// $query_table2 = mysqli_query($connect, "SELECT * FROM `dataset_tablered`");
// while ($d = mysqli_fetch_array($query_table2)) {
// 	$table2[$d['id_kol']][$d['id_bar']] = floatval($d['nilai']);
// }
// //AMBIL DATA LATIH
// $i=0;
// $data_samples=mysqli_query($connect, "SELECT * FROM `dataset`");
// $samples= array();
// while ($d=mysqli_fetch_array($data_samples)) {
// 	$samples['label'][$i]=$d['label'];
// 	$i++;
// }
// $kfold=4;

// $d_dataset = count($table2);
// $t_dataset = count(matrixTranspose($table2));
// // print_r("<hr><h1>SVD DATA LATIH</h1><br>");
// if ($t_dataset > $d_dataset) {
// 	$matrix_dataset = $table2;
// } elseif ( $d_dataset >= $t_dataset) {
// 	$matrix_dataset = matrixTranspose($table2);
// }
// $dataset = new Labeled($matrix_dataset, $samples['label']);
// $folds = $dataset->randomize()->stratifiedFold($kfold);

// for ($i = 0; $i < $kfold; $i++) {
// 	$data[$i] = $folds[$i]->getSamples();
// 	$label[$i] = $folds[$i]->getTargets();
// }
// print_r("<pre>");
// print_r($label);
// // print_r('<pre>DATASET<br>');
// // print_r($dataset);
// for ($i = 0; $i < $kfold; $i++) {
// 	$coba[$i][$i] = $i;
// 	$datalatih[$i] = array_diff_key($data, $coba[$i]);
// 	$labeldatalatih[$i] = array_diff_key($label, $coba[$i]);
// }
// // print_r('<pre>DATA LATIH<br>');
// for ($i = 0; $i < $kfold; $i++) {
// 	$n=0;
// 	foreach ($datalatih[$i] as $key => $value) {
// 		foreach ($value as $kunci => $val) { 
// 			$datalatihbaru[$i][$n] = $datalatih[$i][$key][$kunci] ;
// 			$labeldatalatihbaru[$i][$n] = $labeldatalatih[$i][$key][$kunci];
// 			$n++;
// 		}
// 	}
// 	$datalatihpengujian[$i] = new Labeled($datalatihbaru[$i], $labeldatalatihbaru[$i]);
// }
// // print_r($datalatihbaru);
// // print_r($labeldatalatihbaru);
// // print_r('<pre>DATA LATIH PENGUJIAN KFOLD <br>');
// // print_r($datalatihpengujian);
// // print_r('<pre>DATA UJI PENGUJIAN KFOLD <br>');
// // print_r($folds);

// $report = new MulticlassBreakdown();

// //==========================================================================================================//

// print_r('<pre><h3>PENGUJIAN K-FOLD SVM c=1.5, new Linear(), tol= 1e-3, iterasi=50.0</h3>');
// $estimator = new SVC(1.5, new Linear(), true, 1e-3, 50.0);

// for ($i = 0; $i < $kfold; $i++) {
// 	$estimator->train($datalatihpengujian[$i]);
// 	$predictions[$i] = $estimator->predict($folds[$i]);
// 	$results[$i] = $report->generate($predictions[$i], $folds[$i]->getTargets());
// 	$akurasi[$i] = $results[$i]['overall']['accuracy'];
// 	$fmeasure[$i] = $results[$i]['overall']['f1_score'];
// 	$presisi[$i] = $results[$i]['overall']['precision'];
// 	$recall[$i] = $results[$i]['overall']['recall'];
// 	print_r("<br>AKURASI KE-   " . $i . " = " . $results[$i]['overall']['accuracy']);
// 	print_r("<br>F-MEASURE KE- " . $i . " = " . $results[$i]['overall']['f1_score']);
// 	print_r("<br>PRECISION KE- " . $i . " = " . $results[$i]['overall']['precision']);
// 	print_r("<br>RECALL KE-    " . $i . " = " . $results[$i]['overall']['recall']);
// 	print_r("<br>");
// }
// print_r("<br>RATA-RATA");
// print_r("<br>AKURASI   = " . Stats::mean($akurasi));
// print_r("<br>F-MEASURE = " . Stats::mean($fmeasure));
// print_r("<br>PRESISI   = " . Stats::mean($presisi));
// print_r("<br>RECALL    = " . Stats::mean($recall));
// print_r("<br>");

// //==========================================================================================================//

// arsort($akurasi, SORT_NUMERIC);
// $id_akurasiurut=array_keys($akurasi);
// if($akurasi[$id_akurasiurut[0]] == $akurasi[$id_akurasiurut[1]]){
// 	print_r("<br>");
// 	if ($fmeasure[$id_akurasiurut[0]] >= $fmeasure[$id_akurasiurut[1]]) {
// 		print_r("id paling bagus = " . $id_akurasiurut[$id_akurasiurut[0]] . "<br><br>");
// 		print_r($datalatihbagus = $datalatihpengujian[$id_akurasiurut[$id_akurasiurut[0]]]);
// 		print_r($datujibagus = $folds[$id_akurasiurut[$id_akurasiurut[0]]]);
// 	} else {
// 		print_r("id paling bagus = " . $id_akurasiurut[$id_akurasiurut[1]] . "<br><br>");
// 		print_r($datalatihbagus = $datalatihpengujian[$id_akurasiurut[$id_akurasiurut[1]]]);
// 		print_r($datujibagus = $folds[$id_akurasiurut[$id_akurasiurut[1]]]);
// 	}
// } else {
// 	print_r("<br>");
// 	print_r("id paling bagus = " . $id_akurasiurut[0] . "<br><br>");
// 	print_r($datalatihbagus = $datalatihpengujian[$id_akurasiurut[0]]);
// 	print_r($datujibagus = $folds[$id_akurasiurut[0]]);
// }
// print_r('<br><pre>DATA LATIH PENGUJIAN KFOLD TERBAIK <br>');
// print_r($datalatihbagus);

// $estimator->train($datalatihbagus);
// $estimator->save(dirname(__FILE__) . '/lsasvmbagus.model');
?>