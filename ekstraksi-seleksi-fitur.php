<?php
declare(strict_types=1);
namespace IFVABOT;

require './vendor/autoload.php';

ini_set('memory_limit', '-1');

use Rubix\ML\Tfidf;

include './fungsi.php';
include "./fungsi_svd.php";

$connect = connectDB('ifva');

$stopWords = file_get_contents("term_stopwords_id.txt"); //Panggil File Stopwords 

$stemmerFactory = new \Sastrawi\Stemmer\StemmerFactory(); //Panggil Kelas Stemmer
$stemmer  = $stemmerFactory->createStemmer();

//AMBIL DATA SET
$data_samples=mysqli_query($connect, "SELECT * FROM `dataset`");
$dataset= array();
$i=0;
while ($d=mysqli_fetch_array($data_samples)) {
	$dataset[$i]['pertanyaan'] = $d['pertanyaan'];
	$dataset[$i]['hasil_praproses'] = $d['hasil_praproses'];
	$dataset[$i]['jawaban'] = $d['jawaban'];
	$dataset[$i]['label'] = $d['label'];
	$i++;
}

//AMBIL DATA LATIH
$latih=mysqli_query($connect, "SELECT * FROM `datalatih`");
$data_latih= array();
$i=0;
while ($d=mysqli_fetch_array($latih)) {
	$data_latih[$i]['pertanyaan'] = $d['pertanyaan'];
	$data_latih[$i]['hasil_praproses'] = $d['hasil_praproses'];
	$data_latih[$i]['jawaban'] = $d['jawaban'];
	$data_latih[$i]['label'] = $d['label'];
	$i++;
}

//AMBIL DATA UJI
$uji=mysqli_query($connect, "SELECT * FROM `datauji`");
$data_uji= array();
$i=0;
while ($d=mysqli_fetch_array($uji)) {
	$data_uji[$i]['pertanyaan'] = $d['pertanyaan'];
	$data_uji[$i]['hasil_praproses'] = $d['hasil_praproses'];
	$data_uji[$i]['jawaban'] = $d['jawaban'];
	$data_uji[$i]['label'] = $d['label'];
	$i++;
}
// ======================================================================================================================================== //
print_r("<hr><pre><h1>TF-IDF DATA LATIH</h1>");
foreach ($data_latih as $key => $value) {
	$kalimatstemm_latih[$key] = $value['hasil_praproses'];
	$kalimatarraystemm_latih[$key] = explode(" ", $kalimatstemm_latih[$key]);
}
$m=0;
foreach($kalimatarraystemm_latih as $key => $value ){
	foreach ($value as $kunci => $val){
		$termstemm_latih[$m] = $val;
		$m++;
	}
}
$con=new Tfidf();
$con->proses($kalimatstemm_latih,$termstemm_latih);
$datalatih_table1=$con->table1;
foreach ($datalatih_table1 as $key => $value) {
	$datalatih_term[$key] = array('term'=> $value['term']);
}
foreach ($kalimatarraystemm_latih as $key => $value) {
	$tf_latih[$key] = query($kalimatarraystemm_latih[$key], $datalatih_term);
}
//TF-IDF NORMALISASI L2
$datalatih_table2=normalisasi($con->table2);
//TF DATA UJI
foreach ($data_uji as $key => $value) {
	$kalimatstemm_uji[$key] = $value['hasil_praproses'];
	$kalimatarraystemm_uji[$key] = explode(" ", $kalimatstemm_uji[$key]);
	$tf_uji[$key] = query($kalimatarraystemm_uji[$key], $datalatih_term);
	// $array_uji[$key] = array('tf' => $tf_uji[$key] , 'label' => $labeldatauji[$key]);
}
// print_r($datalatih_table2);
print_r("<hr><pre><h1>TF DATA UJI TERHADAP DATA LATIH</h1>");
// print_r($tf_uji);
//==========================================================================================================================================//
print_r("<hr><h1>SVD TF-IDF DATA LATIH</h1>");
$d_datalatih = count($datalatih_table2);
$t_datalatih = count(matrixTranspose($datalatih_table2));
print_r("jumlah dokumen : ");
print_r($d_datalatih);
print_r("<br>jumlah term : ");
print_r($t_datalatih);
if ($t_datalatih > $d_datalatih) {
	$matrix_datalatih = matrixTranspose($datalatih_table2);
} elseif ( $d_datalatih >= $t_datalatih) {
	$matrix_datalatih = $datalatih_table2;
}
if (!empty($matrix_datalatih)) {
	$svd_datalatih = SVD($matrix_datalatih);
}
print_r("<br>");
// print_r($svd_datalatih);
$ini_k = $svd_datalatih['K'];
$svd_datalatih_red = array();
$svd_datalatih_red['sp'] = matrixConstruct($svd_datalatih['S'], $ini_k, $ini_k);
$svd_datalatih_red['up'] = matrixConstruct($svd_datalatih['U'], count($svd_datalatih['U']), $ini_k);
$svd_datalatih_red['vtp'] = matrixConstruct(matrixTranspose($svd_datalatih['V']), $ini_k, count($svd_datalatih['V']));
$newArr = Array();
foreach ($svd_datalatih_red['sp']  as $key => $val) {
	$r = Array();
	foreach ($val as $v) {
		array_push($r, floatval($v) == 0.0 ? 0 : floatval($v));
	}
	array_push($newArr, $r);
}
$spinvers_datalatih = invert($newArr);
// echo "<hr><pre><h1>LATENT SEMANTIC ANALYSIS DATA LATIH</h1>";
if ($t_datalatih > $d_datalatih) {
	$red = perkalian_matrix($svd_datalatih_red['up'], $spinvers_datalatih);
} elseif ( $d_datalatih >= $t_datalatih) {
	$red = perkalian_matrix(matrixTranspose($svd_datalatih_red['vtp']), $spinvers_datalatih);
}
// REDUKSI PADA DATA LATIH
for ($x=0; $x < $d_datalatih ; $x++) {
	$tf_datalatih_red[$x] = perkalian_matrix(array($tf_latih[$x]), $red);
	foreach ($tf_datalatih_red as $k => $value) {
		foreach ($value as $kunci => $val) {
			$svd_datalatih['red'][$k] = $val;
			$svd_datalatih_normal[$k] = minmax($svd_datalatih['red'][$k]);
		}
	}
}
//REDUKSI PADA DATA UJI
$x=0;
foreach ($data_uji as $k => $value) {
	$tf_uji_red[$k] = perkalian_matrix(array($tf_uji[$k]),$red);
	foreach ($tf_uji_red as $ke => $value) {
		foreach ($value as $kunci => $val) {
			$svd_datauji[$ke] = $val;
			$svd_datauji_normal[$ke] = minmax($svd_datauji[$ke]);
		}
	}
	$array_uji[$k] = array('tf' => $tf_uji[$k] , 'label' => $data_uji[$k]['label'], 'svd_red' => $svd_datauji_normal[$k]);
}
foreach ($data_latih as $key => $value) {
	$labeldatalatih[$key] = $data_latih[$key]['label'];
}
print_r("<hr><h2>REDUKSI DATA LATIH</h2>");
// print_r($svd_datalatih_normal);
print_r("<hr><h2>REDUKSI DATA UJI</h2>");
// print_r($svd_datauji_normal);

// encode array to json array latih
$array_latih = array('term'=> $datalatih_term, 'tf'=> $tf_latih,'tfidf' => $datalatih_table2, 'label' => $labeldatalatih, 'svd_red' => $svd_datalatih_normal, 'red'=>$red);
$json_datalatih = json_encode($array_latih);
//write json to file
if (file_put_contents("json/datalatih.json", $json_datalatih))
	echo "<br><br>JSON DATA LATIH file created successfully...<br>";
else 
	echo "Oops! Error creating json file...";
// encode array to json array uji
$json_datauji = json_encode($array_uji);
//write json to file
if (file_put_contents("json/datauji.json", $json_datauji))
	echo "<br>JSON DATA UJI file created successfully...<br>";
else 
	echo "Oops! Error creating json file...";
// encode array to json array uji
$json_svd_datalatih = json_encode($svd_datalatih);
//write json to file
if (file_put_contents("json/svd.json", $json_svd_datalatih))
	echo "<br>JSON SVD file created successfully...<br>";
else 
	echo "Oops! Error creating json file...";
//........................................................................................................................................//
// print_r("<pre><hr><h1>KANDIDAT RETRIEVAL</h1>");
// $coba = array_values(array_unique($labeldatalatih));
// print_r($coba);
// $x=0;
// foreach ($coba as $k => $value) {
// 	$query_ptyn[$k] = mysqli_query($connect, "SELECT * FROM datalatih WHERE label='$value'");
// 	$kandidat[$k]= array();
// 	$i=0;
// 	while ($d = mysqli_fetch_array($query_ptyn[$k])) {
// 		$kandidat[$k][$i]['id'] = $d['id']-1;
// 		$kandidatstem[$k][$d['id']-1] = $kalimatstemm_latih[$d['id']-1];
// 		$kandidatarraystemm[$k][$d['id']-1] = explode(" ", $kandidatstem[$k][$d['id']-1]);
// 		$i++;
// 	}
// 	$m=0;
// 	foreach($kandidatarraystemm[$k] as $key => $value ){
// 		foreach ($value as $kunci => $val){
// 			$kandidattermstemm[$k][$m] = $val;
// 			$m++;
// 		}
// 	}
// 	$con->proses($kandidatstem[$k], $kandidattermstemm[$k]);
// 	$kandidat_table1[$k]=$con->table1;
// 	foreach ($kandidat_table1[$k] as $key => $value) {
// 		$kandidat_term[$k][$key] = array('term' => $value['term']);
// 	}
// 	foreach ($kandidatarraystemm[$k] as $key => $value) {
// 		$tf_kandidat[$k][$key] = query($kandidatarraystemm[$k][$key], $kandidat_term[$k]);
// 	}
// 	//TF-IDF KANDIDAT NORMALISASI L2
// 	$kandidat_table2[$k]=normalisasi($con->table2);

// 	$d_kandidat[$k] = count($kandidat_table2[$k]);
// 	$t_kandidat[$k] = count(matrixTranspose($kandidat_table2[$k]));
// 	print_r("<pre><h2>SVD DATA KANDIDAT ".$k."</h2>");
// 	print_r("jumlah dokumen : ");
// 	print_r($d_kandidat[$k]);
// 	print_r("<br>jumlah term : ");
// 	print_r($t_kandidat[$k]);
// 	print_r("<br>");

// 	if ($t_kandidat[$k] > $d_kandidat[$k]) {
// 		$matrix_kandidat[$k] = matrixTranspose($kandidat_table2[$k]);
// 	} elseif ( $d_kandidat[$k] >= $t_kandidat[$k]) {
// 		$matrix_kandidat[$k] = $kandidat_table2[$k];
// 	}
// 	if (!empty($matrix_kandidat[$k])) {
// 		$svd_kandidat[$k] = SVD($matrix_kandidat[$k]);
// 	}
// 	$ini_ka[$k] = $svd_kandidat[$k]['K'];
// 	$svd_kandidat_red[$k] = array();
// 	$svd_kandidat_red[$k]['sp'] = matrixConstruct($svd_kandidat[$k]['S'], $ini_ka[$k], $ini_ka[$k]);
// 	$svd_kandidat_red[$k]['up'] = matrixConstruct($svd_kandidat[$k]['U'], count($svd_kandidat[$k]['U']), $ini_ka[$k]);
// 	$svd_kandidat_red[$k]['vtp'] = matrixConstruct(matrixTranspose($svd_kandidat[$k]['V']), $ini_ka[$k], count($svd_kandidat[$k]['V']));
// 	$newArr[$k] = Array();
// 	foreach ($svd_kandidat_red[$k]['sp']  as $key => $val) {
// 		$r = Array();
// 		foreach ($val as $v) {
// 			array_push($r, floatval($v) == 0.0 ? 0 : floatval($v));
// 		}
// 		array_push($newArr[$k], $r);
// 	}
// 	$spinvers_kandidat[$k] = invert($newArr[$k]);

// 	// echo "<hr><pre><h1>LATENT SEMANTIC ANALYSIS DATA KANDIDAT</h1>";
// 	if ($t_kandidat[$k] > $d_kandidat[$k]) {
// 		$red_k[$k] = perkalian_matrix($svd_kandidat_red[$k]['up'], $spinvers_kandidat[$k]);
// 	} elseif ( $d_kandidat >= $t_kandidat) {
// 		$red_k[$k] = perkalian_matrix(matrixTranspose($svd_kandidat_red[$k]['vtp']), $spinvers_kandidat[$k]);
// 	}
// 	// // REDUKSI PADA DATA KANDIDAT
// 	$y=0;
// 	foreach ($tf_kandidat[$k] as $x => $value) {
// 		$tf_kandidat_red[$k][$x] = perkalian_matrix(array($tf_kandidat[$k][$x]), $red_k[$k]);
// 		foreach ($tf_kandidat_red[$k] as $ko => $value) {
// 			foreach ($value as $kunci => $val) {
// 				$svd_kandidat[$k]['red'][$ko] = $val;
// 				$svd_kandidat_normal['red'][$k][$ko] = minmax($svd_kandidat[$k]['red'][$ko]);
// 			}
// 		}
// 		$tfidf_kandidat['tfidf'][$k][$x] = $kandidat_table2[$k][$y];
// 		$y++;
// 	}
// 	print_r($svd_kandidat_normal['red'][$k]);
// 	print_r("<pre><h2>TF-IDF DATA KANDIDAT ".$k."</h2>");
// 	print_r($tfidf_kandidat['tfidf'][$k]);
// 	//TF DATA UJI TERHADAP KANDIDAT
// 	print_r("<pre><h2>TF-IDF DATA UJI TERHADAP KANDIDAT ".$k."</h2>");
// 	foreach ($data_uji as $key => $value) {
// 		$tf_uji_kandidat[$k][$key] = query($kalimatarraystemm_uji[$key], $kandidat_term[$k]);
// 		$tf_uji_kandidat_red[$k][$key] = perkalian_matrix(array($tf_uji_kandidat[$k][$key]),$red_k[$k]);
// 		foreach ($tf_uji_kandidat_red[$k] as $ke => $value) {
// 			foreach ($value as $kunci => $val) {
// 				$svd_datauji_kandidat[$k][$ke] = $val;
// 				$svd_datauji_kandidat_normal[$k][$ke] = minmax($svd_datauji_kandidat[$k][$ke]);
// 			}
// 		}
// 		$array_uji_kandidat[$k][$key] = array('tf' => $tf_uji_kandidat[$k][$key] , 'label' => $data_uji[$key]['label'], 'svd_red' => $svd_datauji_kandidat_normal[$k][$key]);
// 	}
// 	print_r($tf_uji_kandidat);
// 	print_r("<pre><h2>REDUKSI DATA UJI TERHADAP KANDIDAT ".$k."</h2>");
// 	print_r($svd_datauji_kandidat_normal[$k]);
// }

// // encode array to json array latih
// $array_kandidat = array('svd_red' => $svd_kandidat_normal, 'tfidf' => $tfidf_kandidat, 'datauji' => $array_uji_kandidat);
// // print_r($array_kandidat);
// $json_datakandidat = json_encode($array_kandidat);
// //write json to file
// if (file_put_contents("json/datakandidat.json", $json_datakandidat))
// 	echo "JSON DATA KANDIDAT file created successfully...<br>";
// else 
// 	echo "Oops! Error creating json file...";
?>
