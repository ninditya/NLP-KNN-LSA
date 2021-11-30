<?php
declare(strict_types=1);
namespace IFVABOTKNN;

require './vendor/autoload.php';
// ini_set('memory_limit', '1024M'); // or you could use 1G
ini_set('memory_limit', '-1');

use Rubix\ML\ModelManager;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Dataset;

include './fungsi.php';
include './fungsi_svd.php';

$connect = connectDB('ifva');

$stemmerFactory = new \Sastrawi\Stemmer\StemmerFactory(); //Panggil Kelas Stemmer
$stemmer  = $stemmerFactory->createStemmer();

$data_latih = file_get_contents("json/datalatih.json");
$json_latih = json_decode($data_latih, TRUE);
$termlatih = $json_latih['term'];
$labellatih = $json_latih['label'];
$tfidflatih = $json_latih['tfidf'];
$svd = $json_latih['red'];
$svdlatih = $json_latih['svd_red'];

//=========================================================================================================================================//

// DATA BARU
print_r("</pre><hr><pre><h1>PREPROCESSING DATA BARU</h1>");
$pesan = 'email mahasiswa bisa minta siapa?';
$praproses_baru = praproses_baru($pesan);
for ($i=0; $i < count($praproses_baru) ; $i++) { 
	$n = 0;
	$no = 0;
	for ($j=0; $j < count($praproses_baru[$i]) ; $j++) {  
		$stemVal_baru = $stemmer->stem($praproses_baru[$i][$j]);
		$hasilStem_baru[$i][] = $stemVal_baru;
		$stemVal2_baru = $stemmer->stem($praproses_baru[$i][$j]);
		$hasilStem2_baru[$no++] = $stemVal_baru;
		$term_baru[$n++] = $praproses_baru[$i][$j];
	}
	$hasilStemming_baru[$i] = implode(" ", $hasilStem_baru[$i]);
}

print_r("Pesan Asli 	: ". $pesan);
print_r("<br><br>Hasil Preprocessing Data<br><br>");
print_r($hasilStem_baru[0]);

//==============================================================================================================================================//

for ($i=0; $i < count($hasilStem_baru) ; $i++) {
	$tf_q[$i][0] = query($hasilStem_baru[$i], $termlatih);
	$tf_q_red[$i] = perkalian_matrix($tf_q[$i], $svd);
	$tf_q_red_normal[$i] = minmax($tf_q_red[$i][0]);
}
$databaru_red = new Unlabeled($tf_q_red_normal);

$modelManager = new ModelManager(); //memunculkan model latih yang disimpan
echo "<hr><pre><h1>PREDIKSI K Nearest Neighbors</h1>";
$model = $modelManager->restoreFromFile(__DIR__.'/knn-lsa.model');
$predictions = $model->predict($databaru_red);

print_r("PREDIKSI KNN-LSA = ");
print_r($predictions[0]);

//==============================================================================================================================================//
//AMBIL DATA LATIH
$i=0;
// print_r("<hr><h1>KANDIDAT JAWABAN </h1>");
$query_ptyn = mysqli_query($connect, "SELECT * FROM datalatih WHERE label='$predictions[0]'");
$kandidat= array();
while ($d = mysqli_fetch_array($query_ptyn)) {
	$kandidat[$d['id']-1]['pertanyaan'] = $d['pertanyaan'];
	$kandidat[$d['id']-1]['hasil_praproses'] = $d['hasil_praproses'];
	$kandidat[$d['id']-1]['jawaban'] = $d['jawaban'];
	$kandidat[$d['id']-1]['label'] = $d['label'];
	$i++;
}
$cek = array_flip(array_values(array_unique($labellatih)));
$alamat = $cek[$predictions[0]];
$json_kandidat = file_get_contents("json/datakandidat.json");
$data_kandidat = json_decode($json_kandidat, TRUE);
$tfidfkandidat = $data_kandidat['tfidf']['tfidf'][$alamat];
$svdkandidat = $data_kandidat['svd_red']['red'][$alamat];
$red_k = $data_kandidat['red_k'][$alamat];
$termkandidat = $data_kandidat['term'][$alamat];
$tf_q_kandidat[0] = query($hasilStem_baru[0], $termkandidat);
$tf_q_kandidat_red = perkalian_matrix($tf_q_kandidat, $red_k);
$tf_q_kandidat_red_normal[0] = minmax($tf_q_kandidat_red[0]);

//echo "<hr><pre><h1> COSINE SIMILARITY DATA BARU DAN KANDIDAT JAWABAN</h1>";
foreach ($tfidfkandidat as $key => $value) {
	$nilaicossim[$key]= cosinesimilarity($tf_q_kandidat,matrixTranspose(array($value)));
	foreach ($nilaicossim[$key] as $kunci => $val) {
		$sorted[$key] = $val[0];
		arsort($sorted, SORT_NUMERIC);
	}
	// $nilaicossimlsa[$key]= cosinesimilarity($tf_q_kandidat_red_normal,matrixTranspose(array($svdkandidat[$key])));
	// foreach ($nilaicossimlsa[$key] as $kunci => $val) {
	// 	$sortedlsa[$key] = $val[0];
	// 	arsort($sortedlsa, SORT_NUMERIC);
	// }
}
// print_r($sortedlsa);
//=====================================================================
print_r("<h1>KANDIDAT JAWABAN VSM</h1>");
$coba = 0;
foreach ($sorted as $key => $value) {
	$gabung_idx[$coba]['idx_pertanyaan']=$key;
	$gabung_idx[$coba]['pertanyaan']= $kandidat[$key]['pertanyaan'];
	$gabung_idx[$coba]['hasil_praproses']=$kandidat[$key]['hasil_praproses'];
	$gabung_idx[$coba]['jawaban']=$kandidat[$key]['jawaban'];
	$gabung_idx[$coba]['label']=$kandidat[$key]['label'];
	$gabung_idx[$coba]['nilaicossim']=$sorted[$key];
	$coba++;
}
print_r("<pre>");
print_r($gabung_idx);
// ======================================================================
print_r("<hr><h1>EKSTRAKSI JAWABAN VSM </h1>");
// echo "<hr><pre><h1> EKSTRAKSI JAWABAN VSM </h1>";

$nilai_max = $gabung_idx[0]['nilaicossim'];
$jawabansalah = "Maaf saya belum punya jawaban pertanyaanmu.";

$gabung_baru = Array();
$gabung_baru[0]['pertanyaan'] = $pesan;
$gabung_baru[0]['hasil_praproses'] = $hasilStemming_baru[0];
if ($nilai_max != 0) {
	$id_max = array_search($nilai_max, $gabung_idx);
	// print_r("nilai_max : ");
	// print_r($nilai_max);
	// print_r("<br>id_max : ");
	// print_r($gabung_idx[0]['idx_pertanyaan']);
	// print_r("<br>jawaban : ");
	// print_r($gabung_idx[0]['jawaban']);
	$gabung_baru[0]['jawaban'] = $gabung_idx[0]['jawaban'];
} else {
	print_r("<br>jawaban : ");
	print_r($jawabansalah);
	$gabung_baru[0]['jawaban'] = $jawabansalah;
}
$gabung_baru[0]['label'] = $predictions[0];
print_r("<pre>");
print_r($gabung_baru[0]);

foreach ($gabung_baru as $row) {
	mysqli_query($connect, "INSERT INTO pertanyaan_paling_baru ( id, pertanyaan, jawaban, label) VALUES ('','$row[pertanyaan]','$row[jawaban]','$row[label]')");
}
// //====================================================================//
// print_r("<h1>KANDIDAT JAWABAN LSA</h1>");
// $c = 0;
// foreach ($sortedlsa as $key => $value) {
// 	$gabung_lsa[$c]['idx_pertanyaan']=$key;
// 	$gabung_lsa[$c]['pertanyaan']= $kandidat[$key]['pertanyaan'];
// 	$gabung_lsa[$c]['hasil_praproses']=$kandidat[$key]['hasil_praproses'];
// 	$gabung_lsa[$c]['jawaban']=$kandidat[$key]['jawaban'];
// 	$gabung_lsa[$c]['label']=$kandidat[$key]['label'];
// 	$gabung_lsa[$c]['nilaicossim']=$sortedlsa[$key];
// 	$c++;
// }
// print_r("<pre>");
// print_r($gabung_lsa);
// //====================================================================//
// print_r("<hr><h1>EKSTRAKSI JAWABAN LSA </h1>");
// // echo "<hr><pre><h1> EKSTRAKSI JAWABAN VSM </h1>";

// $nilai_max_lsa = $gabung_lsa[0]['nilaicossim'];
// $jawabansalah_lsa = "Maaf saya belum punya jawaban pertanyaanmu.";

// $gabung_baru_lsa = Array();
// $gabung_baru_lsa[0]['pertanyaan'] = $pesan;
// $gabung_baru_lsa[0]['hasil_praproses'] = $hasilStemming_baru[0];
// if ($nilai_max_lsa != 0) {
// 	$id_max_lsa = array_search($nilai_max_lsa, $gabung_lsa);
// 	// print_r("nilai_max : ");
// 	// print_r($nilai_max);
// 	// print_r("<br>id_max : ");
// 	// print_r($gabung_lsa[0]['lsa_pertanyaan']);
// 	// print_r("<br>jawaban : ");
// 	// print_r($gabung_lsa[0]['jawaban']);
// 	$gabung_baru_lsa[0]['jawaban'] = $gabung_lsa[0]['jawaban'];
// } else {
// 	print_r("<br>jawaban : ");
// 	print_r($jawabansalah_lsa);
// 	$gabung_baru_lsa[0]['jawaban'] = $jawabansalah_lsa;
// }
// $gabung_baru_lsa[0]['label'] = $predictions[0];
// print_r("<pre>");
// print_r($gabung_baru_lsa[0]);

// foreach ($gabung_baru_lsa as $row) {
// 	mysqli_query($connect, "INSERT INTO pertanyaan_paling_baru ( id, pertanyaan, jawaban, label) VALUES ('','$row[pertanyaan]','$row[jawaban]','$row[label]')");
// }

?>