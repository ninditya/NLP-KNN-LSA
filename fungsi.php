<?php
function connectDB ($db){
	$server = "localhost";
	$username = "root";
	$password = "";
	$connect = mysqli_connect($server,$username,$password,$db);
	if (!$connect) {
		die('Could not connect: ' . mysqli_error());
	}
	return $connect;
}

function praproses($input){
	if (!empty($input)) {
		//panggil fungsi prepos trainning pertanyaan
		$stopWords = file_get_contents("term_stopwords_id.txt"); //Panggil File Stopwords 
		$pisahKalimat = pecahkalimat($input); 
		$caseFolding = caseFolding($pisahKalimat);
		$tokenizing = tokenizing($caseFolding);
		$stopwordsRemoved = stopwordsRemoval($stopWords,$tokenizing);
	}
	return $stopwordsRemoved;
}

function praproses_baru($input){
	if (!empty($input)) {
		//panggil fungsi prepos trainning pertanyaan
		$stopWords = file_get_contents("term_stopwords_id.txt"); //Panggil File Stopwords 
		$pisahKalimat[0] = pecahkalimat3($input); 
		$caseFolding = caseFolding($pisahKalimat);
		$tokenizing = tokenizing($caseFolding);
		$stopwordsRemoved = stopwordsRemoval($stopWords,$tokenizing);
	}
	return $stopwordsRemoved;
}

function pecahkalimat($input){
	//split konten per kalimat
	$pisahKalimat = array();
	$hasilPisah = array();
	// $teks = str_replace(".", " ", $input);
	// $pisahKalimat = preg_split("/[.!?]+/", $teks);
	$pisahKalimat = preg_split("/[.!?]+/", $input);
	//$pisahKalimat = preg_split("#\n#", $input);
	$pisahKalimat = array_slice($pisahKalimat, 0, sizeof($pisahKalimat)-1); // buang array terakhir (kosong)
	for ($i=0; $i < count($pisahKalimat); $i++) { 
		array_push($hasilPisah, $pisahKalimat[$i]);
	}
	return $pisahKalimat;

}

function pecahkalimat2($input){
	foreach ($input as $key => $value) {
		$pisahKalimat = preg_replace("/[.!?]+/"," ", $value);
		$input[$key] = $pisahKalimat;
	}
	return $input;
}

function pecahkalimat3($input){
	$pisahKalimat = preg_replace("/[.!?]+/"," ", $input);
	return $pisahKalimat;
}

function caseFolding($pisahKalimat){
	$caseFolding = array();
	$caseFolding = array_map("strtolower", $pisahKalimat);
	$caseFolding = preg_replace("/[\d\W]+/"," ", $caseFolding); //[\d] hapus angka [\W] hapus simbol
	return $caseFolding;
}

function tokenizing($caseFolding){
	$tokenizing = array();
	$hasilTokenizing = array();
	for ($i=0; $i <count($caseFolding) ; $i++) { 
		$tokenizing = preg_split("/[\s]+/", $caseFolding[$i]);
		array_push($hasilTokenizing, $tokenizing);
	}
	//rapih rapih array
	$hasilTokenizing = array_map('array_filter', $hasilTokenizing);
	$hasilTokenizing = array_filter($hasilTokenizing);
	$hasilTokenizing = array_map('array_values', $hasilTokenizing);
	$hasilTokenizing = array_values($hasilTokenizing);
	return $hasilTokenizing;
}

function stopwordsRemoval($stopWords,$tokenizing){
	//pisah berdasarkan spasi
	$stopwordsRemoved = array();
	$hasilStopWordsRemoved = array();
	$getstopWords = preg_split("/[\s]+/", $stopWords);
	for ($i=0; $i < count($tokenizing); $i++) { 
		$stopwordsRemoved = array_diff($tokenizing[$i], $getstopWords);	
		$stopwordsRemoved = array_values($stopwordsRemoved); // perbaiki indeks
		array_push($hasilStopWordsRemoved, $stopwordsRemoved);
	}
	//rapih rapih array
	$hasilStopWordsRemoved = array_map('array_filter', $hasilStopWordsRemoved);
	$hasilStopWordsRemoved = array_filter($hasilStopWordsRemoved);
	$hasilStopWordsRemoved = array_map('array_values', $hasilStopWordsRemoved);
	$hasilStopWordsRemoved = array_values($hasilStopWordsRemoved);
	return $hasilStopWordsRemoved;
}

function textpreprocessing($isi) {
	$HasilstopwordsRemoved = array();
	//panggil fungsi prepos
	$pisahKalimat = pecahkalimat($isi); 
	$caseFolding = caseFolding($pisahKalimat);
	$tokenizing = tokenizing($caseFolding);
	$HasilstopwordsRemoved = stopwordsRemoval($stopWords,$tokenizing);
	return $HasilstopwordsRemoved;
}

function getFrequency($hasilStem){
	$frequency = []; //array baru untuk tampung
	foreach ($hasilStem as $key => $value) {
		foreach ($value as $key2 => $value2) {
			$needle = $value2; //key yang di cari
			if (array_key_exists($needle, $frequency))
				$frequency[$needle]++;
			else
				$frequency[$needle] = 1;
		}
	}
	return $frequency;
}	

function getDF($hasilStem){
	$df = [];
	$unique = array_map('array_unique', $hasilStem);
	foreach ($unique as $key => $value) {
		foreach ($value as $key2 => $value2) {
			if(array_key_exists($value2, $df))
				$df[$value2]++;
			else
				$df[$value2] = 1;
		}
	}
	return $df;
}

function daftarKata($w){
	$daftarKata = [];
	foreach ($w as $key => $value) {
		$daftarKata[] = $key;
	}
	return $daftarKata;
}

function createDummyMatrix($daftarKata,$hasilStem){
	$matrixDummy = [];
	for ($i=0; $i < count($hasilStem) ; $i++) { 
		for ($j=0; $j < count($daftarKata) ; $j++) { 
			$matrixDummy[$i][] = $daftarKata[$j];
		}
	}
	for ($i=0; $i < count($matrixDummy); $i++) { 
		for ($j=0; $j < count($matrixDummy[$i]) ; $j++) { 
			if (!in_array($matrixDummy[$i][$j], $hasilStem[$i])) {
				$matrixDummy[$i][$j] = 0;
			}
		}
	}
	return $matrixDummy;
}

function createMatrix($matrixDummy,$w,$daftarKata){
	$matrix = [];
	foreach ($matrixDummy as $key => $value) {
		foreach ($value as $key2 => $value2) {
			$dummy = $matrixDummy[$key][$key2];
			if (in_array($dummy, $daftarKata))
				$matrix[$key][$key2] = $w[$dummy];
			//ubah null jadi 0
			if (is_null($matrix[$key][$key2]))
				$matrix[$key][$key2] = 0;
		}
	}
	return $matrix;
}



?>	