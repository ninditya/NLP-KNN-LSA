<?php 
function threshold($threshold, $nilaicossim) {
    $nilaithreshold = array();
    foreach ($nilaicossim as $id_cossim => $nilai) {
        if ($nilai > $threshold) {
            $nilaithreshold[$id_cossim+1] = $nilai;
        }
    }
    return $nilaithreshold;
}

function matrixRound($matrix){
    $m = count($matrix);
    $n = count($matrix[0]);

    for($i = 0; $i < $m; $i++){
        for($j = 0; $j < $n; $j++){
            $matrixT[$i][$j] = round($matrix[$i][$j], 2);
        }
    }
    return $matrixT;
}

function matrixTranspose($matrix){
    $m = count($matrix);
    $n = count($matrix[0]);
    for($i = 0; $i < $n; $i++){
        for($j = 0; $j < $m; $j++){
            $matrixT[$i][$j] = $matrix[$j][$i];
        }
    }
    return $matrixT;
}
function Transpose($matrix){
    $m = count($matrix);
    $n = count(array($matrix[0]));
    for($i = 0; $i < $n; $i++){
        for($j = 0; $j < $m; $j++){
            $matrixT[$i][$j] = $matrix[$j][$i];
        }
    }
    return $matrixT;
}
function pythag($a, $b){
    $absa = abs($a);
    $absb = abs($b);
    if ($absa > $absb) {
        return $absa * sqrt(1.0 + pow($absb / $absa, 2));
    } elseif ($absb > 0.0) {
        return $absb * sqrt(1.0 + pow($absa / $absb, 2));
    } else {
        return 0.0;
    }
    // if( $absa > $absb ){
    //     return $absa * sqrt( 1.0 + pow( $absb / $absa , 2) );
    // }else {
    //     if( $absb > 0.0 ){
    //         return $absb * sqrt( 1.0 + pow( $absa / $absb, 2 ) );
    //     }else {
    //         return 0.0;
    //     }
    // }
}
function maximum($a, $b){

    if($a < $b){
        return $b;
    }else {
        return $a;
    }
}
/**
* minimum
* 
* @param integer $a
* @param integer $b
* @return integer
*/
function minimum($a, $b){
    if($a > $b){
        return $b;
    }else {
        return $a;
    }
}

function sameSign($a, $b){
    if($b >= 0){
        $result = abs($a);
    }else {
        $result = -abs($a);
    }
    return $result;
}

function matrixConstruct($matrix, $rows, $columns){
    for($i = 0; $i < $rows; $i++){
        for($j = 0; $j < $columns; $j++){
            $neoMatrix[$i][$j] = $matrix[$i][$j];
        }
    }
    return $neoMatrix;
}

function SVD($matrix){
    $m = count($matrix);
    $n = count($matrix[0]);

    $U  = matrixConstruct($matrix, $m, $n);
    $V  = matrixConstruct($matrix, $n, $n);

    $w = [];
    $eps = 2.22045e-016;

    // Decompose Phase

    // Householder reduction to bidiagonal form.
    $g = $scale = $anorm = 0.0;
    $l=0;
    $rvl = [];
    for($i = 0; $i < $n; $i++){
        $l = $i + 2;
        $rv1[$i] = $scale * $g;
        $g = $s = $scale = 0.0;
        if($i < $m){
            for($k = $i; $k < $m; $k++)
                $scale += abs($U[$k][$i]);
            if($scale != 0.0) {
                for($k = $i; $k < $m; $k++) {
                    $U[$k][$i] /= $scale;
                    $s += $U[$k][$i] * $U[$k][$i];
                }
                $f = $U[$i][$i];
                $g = -sameSign(sqrt($s), $f);
                $h = $f * $g - $s;
                $U[$i][$i] = $f - $g;
                for($j = $l - 1; $j < $n; $j++){
                    for($s = 0.0, $k = $i; $k < $m; $k++) $s += $U[$k][$i] * $U[$k][$j];
                    $f = $s / $h;
                    for($k = $i; $k < $m; $k++) $U[$k][$j] += $f * $U[$k][$i];
                }
                for($k = $i; $k < $m; $k++) $U[$k][$i] *= $scale;
            }
        }
        $W[$i] = $scale * $g;
        $g = $s = $scale = 0.0;
        if($i + 1 <= $m && $i + 1 != $n){
            for ($k= $l - 1; $k < $n; $k++) $scale += abs($U[$i][$k]);
            if($scale != 0.0){
                for ($k= $l - 1; $k < $n; $k++){
                    $U[$i][$k] /= $scale;
                    $s += $U[$i][$k] * $U[$i][$k];
                }
                $f = $U[$i][$l - 1];
                $g = -sameSign(sqrt($s), $f);
                $h = $f * $g - $s;
                $U[$i][$l - 1] = $f - $g;
                for($k = $l - 1; $k < $n; $k++) $rv1[$k] = $U[$i][$k] / $h;
                for($j = $l - 1; $j < $m; $j++){
                    for($s = 0.0, $k = $l - 1; $k < $n; $k++) $s += $U[$j][$k] * $U[$i][$k];
                    for($k = $l - 1; $k < $n; $k++) $U[$j][$k] += $s * $rv1[$k];
                }
                for($k= $l - 1; $k < $n; $k++) $U[$i][$k] *= $scale;
            }
        }
        $anorm = max($anorm, (abs($W[$i])+abs($rv1[$i])));
    }
// Accumulation of right-hand transformations.
    for($i = $n - 1; $i >= 0; $i--){
        if($i < $n - 1){
            if($g != 0.0){
                    for($j = $l; $j < $n; $j++) // Double division to avoid possible underflow.
                    $V[$j][$i] = ($U[$i][$j] / $U[$i][$l]) / $g;
                    for($j = $l; $j < $n; $j++){
                        for($s = 0.0, $k = $l; $k < $n; $k++) $s += ($U[$i][$k] * $V[$k][$j]);
                        for($k = $l; $k < $n; $k++) $V[$k][$j] += $s * $V[$k][$i];
                    }
                }
                for($j = $l; $j < $n; $j++) $V[$i][$j] = $V[$j][$i] = 0.0;
            }
            $V[$i][$i] = 1.0;
            $g = $rv1[$i];
            $l = $i;
        }
    // Accumulation of left-hand transformations.
        for($i = minimum($m, $n) - 1; $i >= 0; $i--){
            $l = $i + 1;
            $g = $W[$i];
            for($j = $l; $j < $n; $j++) $U[$i][$j] = 0.0;
            if($g != 0.0){
                $g = 1.0 / $g;
                for($j = $l; $j < $n; $j++){
                    for($s = 0.0, $k = $l; $k < $m; $k++) $s += $U[$k][$i] * $U[$k][$j];
                    $f = ($s / $U[$i][$i]) * $g;
                    for($k = $i; $k < $m; $k++) $U[$k][$j] += $f * $U[$k][$i];
                }
                for($j = $i; $j < $m; $j++) $U[$j][$i] *= $g;
            }else {
                for($j = $i; $j < $m; $j++) $U[$j][$i] = 0.0;
            }
            ++$U[$i][$i];
        }
    // Diagonalization of the bidiagonal form
    // Loop over singular values, and over allowed iterations.
        $nm=0;
        for($k = $n - 1; $k >= 0; $k--){
            for($its = 0; $its < 30; $its++){
                $flag = true;
                for($l = $k; $l >= 0; $l--){
                    $nm = $l - 1;
                    if( $l == 0 || abs($rv1[$l]) <= $eps*$anorm){
                        $flag = false;
                        break;
                    }
                    if(abs($W[$nm]) <= $eps*$anorm) break;
                }
                if($flag){
                    $c = 0.0;  // Cancellation of rv1[l], if l > 0.
                    $s = 1.0;
                    for($i = $l; $i < $k + 1; $i++){
                        $f = $s * $rv1[$i];
                        $rv1[$i] = $c * $rv1[$i];
                        if(abs($f) <= $eps*$anorm) break;
                        $g = $W[$i];
                        $h = pythag($f,$g);
                        $W[$i] = $h;
                        $h = 1.0 / $h;
                        $c = $g * $h;
                        $s = -$f * $h;
                        for($j = 0; $j < $m; $j++){
                            $y = $U[$j][$nm];
                            $z = $U[$j][$i];
                            $U[$j][$nm] = $y * $c + $z * $s;
                            $U[$j][$i] = $z * $c - $y * $s;
                        }
                    }
                }
                $z = $W[$k];
                if($l == $k){
                    if($z < 0.0){
                        $W[$k] = -$z; // Singular value is made nonnegative.
                        for($j = 0; $j < $n; $j++) $V[$j][$k] = -$V[$j][$k];
                    }
                    break;
                }
                if($its == 29) print("no convergence in 30 svd iterations");
                $x = $W[$l]; // Shift from bottom 2-by-2 minor.
                $nm = $k - 1;
                $y = $W[$nm];
                $g = $rv1[$nm];
                $h = $rv1[$k];
                $f = (($y - $z) * ($y + $z) + ($g - $h) * ($g + $h)) / (2.0 * $h * $y);
                $g = pythag($f,1.0);
                $f = (($x - $z) * ($x + $z) + $h * (($y / ($f + sameSign($g,$f))) - $h)) / $x;
                $c = $s = 1.0;
                for($j = $l; $j <= $nm; $j++){
                    $i = $j + 1;
                    $g = $rv1[$i];
                    $y = $W[$i];
                    $h = $s * $g;
                    $g = $c * $g;
                    $z = pythag($f,$h);
                    $rv1[$j] = $z;
                    $c = $f / $z;
                    $s = $h / $z;
                    $f = $x * $c + $g * $s;
                    $g = $g * $c - $x * $s;
                    $h = $y * $s;
                    $y *= $c;
                    for($jj = 0; $jj < $n; $jj++){
                        $x = $V[$jj][$j];
                        $z = $V[$jj][$i];
                        $V[$jj][$j] = $x * $c + $z * $s;
                        $V[$jj][$i] = $z * $c - $x * $s;
                    }
                    $z = pythag($f,$h);
                    $W[$j] = $z;  // Rotation can be arbitrary if z = 0.
                    if($z){
                        $z = 1.0 / $z;
                        $c = $f * $z;
                        $s = $h * $z;
                    }
                    $f = $c * $g + $s * $y;
                    $x = $c * $y - $s * $g;
                    for($jj = 0; $jj < $m; $jj++){
                        $y = $U[$jj][$j];
                        $z = $U[$jj][$i];
                        $U[$jj][$j] = $y * $c + $z * $s;
                        $U[$jj][$i] = $z * $c - $y * $s;
                    }
                }
                $rv1[$l] = 0.0;
                $rv1[$k] = $f;
                $W[$k] = $x;
            }
        }     
        // Reorder Phase
        // Sort. The method is Shell's sort.
        // (The work is negligible as compared to that already done in decompose phase.)
        $inc = 1;
        do {
            $inc *= 3;
            $inc++;
        }   while($inc <= $n);
        
        $su = [];
        $sv = [];
        do {
            $inc /= 3;
            for($i = $inc; $i < $n; $i++){
                $sw = $W[$i];
                for($k = 0; $k < $m; $k++) $su[$k] = $U[$k][$i];
                for($k = 0; $k < $n; $k++) $sv[$k] = $V[$k][$i];
                $j = $i;
                while($W[$j - $inc] < $sw){
                    $W[$j] = $W[$j - $inc];
                    for($k = 0; $k < $m; $k++) $U[$k][$j] = $U[$k][$j - $inc];
                    for($k = 0; $k < $n; $k++) $V[$k][$j] = $V[$k][$j - $inc];
                    $j -= $inc;
                    if($j < $inc) break;
                }
                $W[$j] = $sw;
                for($k = 0; $k < $m; $k++) $U[$k][$j] = $su[$k];
                for($k = 0; $k < $n; $k++) $V[$k][$j] = $sv[$k];
            }
        }  while($inc > 1);

        for($k = 0; $k < $n; $k++){
            $s = 0;
            for($i = 0; $i < $m; $i++) if ($U[$i][$k] < 0.0) $s++;
            for($j = 0; $j < $n; $j++) if ($V[$j][$k] < 0.0) $s++;
            if($s > ($m + $n)/2) {
                for($i = 0; $i < $m; $i++) $U[$i][$k] = - $U[$i][$k];
                for($j = 0; $j < $n; $j++) $V[$j][$k] = - $V[$j][$k];
            }
        }
        // //tadinya di delete
        // // calculate the rank
        // $rank = 0;
        // for($i = 0; $i < count($W); $i++){
        //     if(round($W[$i], 4) > 0){
        //         $rank += 1;
        //     }
        // }
        // // Low-Rank Approximation
        // $q = 0.9;
        // $k = 0;
        // for($i = 0; $i < $rank; $i++) $frobA += $W[$i];
        // do{
        //     for($i = 0; $i <= $k; $i++) $frobAk += $W[$i];
        //     $clt = $frobAk / $frobA;
        //     $k++;
        // }   while($clt < $q);
        // //tadinya di delete
        // prepare S matrix as n*n daigonal matrix of singular values
        for($i = 0; $i < $n; $i++){
            for($j = 0; $j < $n; $j++){
                $S[$i][$j] = 0;
                $S[$i][$i] = $W[$i];
            }
        }
        $matrices['U'] = $U;
        $matrices['S'] = $S;
        $matrices['W'] = $W;
        $matrices['V'] = matrixTranspose($V);
        //$matrices['Rank'] = $rank;
        $matrices['K'] = $k;
        return $matrices;
    }
    //MATRIKS INVERS
    function invert($A, $debug = FALSE)  {
        /// @todo check rows = columns
        $n = count($A);
        // get and append identity matrix
        $I = identity_matrix($n);
        for ($i = 0; $i < $n; ++ $i) {
            $A[$i] = array_merge($A[$i], $I[$i]);
        }
        if ($debug) {
          echo "\nStarting matrix: ";
          print_matrix($A);
      }
        // forward run
      for ($j = 0; $j < $n-1; ++ $j) {
    // for all remaining rows (diagonally)
          for ($i = $j+1; $i < $n; ++ $i) {
    // if the value is not already 0
            if ($A[$i][$j] !== 0) {
    // adjust scale to pivot row
    // subtract pivot row from current
                print_r($A[$j][$j]);
                $scalar = $A[$j][$j] / $A[$i][$j];
                for ($jj = $j; $jj < $n*2; ++ $jj) {
                    $A[$i][$jj] *= $scalar;
                    $A[$i][$jj] -= $A[$j][$jj];
                }
            }
        }
        if ($debug) {
            echo "\nForward iteration $j: ";
            print_matrix($A);
        }
    }
    // reverse run
    for ($j = $n-1; $j > 0; -- $j) {
        for ($i = $j-1; $i >= 0; -- $i) {
            if ($A[$i][$j] !== 0) {
                $scalar = $A[$j][$j] / $A[$i][$j];
                for ($jj = $i; $jj < $n*2; ++ $jj) {
                    $A[$i][$jj] *= $scalar;
                    $A[$i][$jj] -= $A[$j][$jj];
                }
            }
        }
        if ($debug) {
            echo "\nReverse iteration $j: ";
            print_matrix($A);
        }
    }
                    // last run to make all diagonal 1s
                    // @note this can be done in last iteration (i.e. reverse run) too!
    for ($j = 0; $j < $n; ++ $j) {
        if ($A[$j][$j] !== 1) {
                            // $scalar = 1 / $A[$j][$j];
            $scalar = ($A[$j][$j])!= 0 ? 1 / $A[$j][$j] : 1;
            for ($jj = $j; $jj < $n*2; ++ $jj) {
                $A[$j][$jj] *= $scalar;
            }
        }
        if ($debug) {
            echo "\n1-out iteration $j: ";
            print_matrix($A);
        }
    }
                    //take out the matrix inverse to return
    $Inv = array();
    for ($i = 0; $i < $n; ++ $i) {
        $Inv[$i] = array_slice($A[$i], $n);
    }
    return $Inv;
}
/**
     * Prints matrix
     *
     * @param array $A matrix
     * @param integer $decimals number of decimals
     */
function print_matrix($A, $decimals = 6)
{
    echo "<pre>";
    foreach ($A as $row) {

        foreach ($row as $i) {
            echo "\t" . sprintf("%01.{$decimals}f", round($i, $decimals));
        }
        echo "\t]";
        print("\n");
    }
}
    /**
     * Produces an identity matrix of given size
     *
     * @param integer $n size of identity matrix
     *
     * @return array identity matrix
     */
    function identity_matrix($n)
    {
        $I = array();
        for ($i = 0; $i < $n; ++ $i) {
            for ($j = 0; $j < $n; ++ $j) {
                $I[$i][$j] = ($i == $j) ? 1 : 0;
            }
        }
        return $I;
    }

    function perkalian_matrix($m1, $m2) {
        // checking if the matrices can be multiplied
        $rows_m1 = count($m1);
        $cols_m1 = count($m1[0]);
        $rows_m2 = count($m2);
        $cols_m2 = count($m2[0]);
        if ($cols_m1 != $rows_m2) {
            echo ('The matrices cannot be multiplied!');
            die();
        }
        $prod = [];

        for ($i=0; $i<$rows_m1; $i++) {
            for ($j=0; $j<$cols_m2; $j++) {
                $prod[$i][$j] = 0;
                for ($k=0; $k<$rows_m2; $k++) {
                    $prod[$i][$j] += $m1[$i][$k] * $m2[$k][$j];
                }
            }
        }

        return $prod;
    }


    // fungsi term qeury
    function uji($query, $term) {
        $index = 0;
        $df_q = array();
        foreach ($term as $kunci) {
            $p=0;
            foreach ($query as $key => $value ) {
                if ($term[$index] == $value) {
                    $p++;
                }
            }
            array_push($df_q, $p);
            $index++;
        }
        return $df_q;
    }

    // fungsi term qeury
    function query($query, $term) {
        $index = 1;
        $df_q = array();
        foreach ($term as $kunci) {
            $p=0;
            foreach ($query as $key => $value) {
                if ($kunci['term'] == $value) {
                    $p++;
                // echo "ini masuk lho";
                }
                // echo "ini ga masuk lho";
            // $df_q[$index] = $p;
            }
            array_push($df_q, $p);
            $index++;
        }
        return $df_q;
    }

    function query2($query, $term) {
        $index = 1;
        $df_q2 = array();
        foreach ($term as $kunci) {
            $p=0;
            foreach ($query as $key => $value) {
                if ($kunci == $value) {
                    $p++;
                // echo "ini masuk lho";
                }
                // echo "ini ga masuk lho";
            // $df_q[$index] = $p;
            }
            array_push($df_q2, $p);
            $index++;
        }
        return $df_q2;
    }
    //COSINE SIMILARITY
    function cosinesimilaritylsa($matriks_a,$matriks_b)
    {
        $nilaicossim = array();
        $bawah_q = 0;
        $bawah_d = 0;
        for ($i=0; $i < sizeof($matriks_a); $i++) { 
            for ($k=0; $k < sizeof($matriks_a[0]); $k++) { 
                $bawah_q += $matriks_a[$i][$k]**2 ;    
            }    
        }
        for ($i=0; $i<sizeof($matriks_a); $i++) {
            for ($j=0; $j<sizeof($matriks_b[0]); $j++) {
                $temp = 0;
                $atas = 0;
                $bawah = 0;
                for ($k=0; $k<sizeof($matriks_b); $k++) {
                    // $temp += array_sum(array($matriks_a[$i][$k] * $matriks_b[$k][$j])) / 
                    // (sqrt($matriks_a[$i][$k] * $matriks_b[$k][$j]) * sqrt($matriks_a[$i][$k] * $matriks_b[$k][$j]));
                    // if($temp < 0)
                    // {
                    //     $temp *= -1;
                    // }
                    $atas += ($matriks_a[$i][$k] * $matriks_b[$k][$j]);
                    $bawah_d += $matriks_b[$k][$j]**2 ;
                }
                // $temp = $atas/(sqrt($bawah_q)*sqrt($bawah_d));
                $temp = (sqrt($bawah_q)*sqrt($bawah_d))!=0 ? ($atas/(sqrt($bawah_q)*sqrt($bawah_d)))+1/2 : 0 ;
                $nilaicossim[$i][$j] = $temp;
                // $nilaicossim[$i][$j] = $temp;
            }
        }
        return $nilaicossim;
    }

    function cosinesimilarity($matriks_a,$matriks_b)
    {
        $nilaicossim = array();
        $bawah_q = 0;
        $bawah_d = 0;
        for ($i=0; $i < sizeof($matriks_a); $i++) { 
            for ($k=0; $k < sizeof($matriks_a[0]); $k++) { 
                $bawah_q += $matriks_a[$i][$k]**2 ;    
            }    
        }
        for ($i=0; $i<sizeof($matriks_a); $i++) {
            for ($j=0; $j<sizeof($matriks_b[0]); $j++) {
                $temp = 0;
                $atas = 0;
                $bawah = 0;
                for ($k=0; $k<sizeof($matriks_b); $k++) {
                    // $temp += array_sum(array($matriks_a[$i][$k] * $matriks_b[$k][$j])) / 
                    // (sqrt($matriks_a[$i][$k] * $matriks_b[$k][$j]) * sqrt($matriks_a[$i][$k] * $matriks_b[$k][$j]));
                    // if($temp < 0)
                    // {
                    //     $temp *= -1;
                    // }
                    $atas += ($matriks_a[$i][$k] * $matriks_b[$k][$j]);
                    $bawah_d += $matriks_b[$k][$j]**2 ;
                }
                $temp = (sqrt($bawah_q)*sqrt($bawah_d))!=0 ? $atas/(sqrt($bawah_q)*sqrt($bawah_d)) : 0 ;
                // $temp = round($atas/(sqrt($bawah_q)*sqrt($bawah_d)));
                $nilaicossim[$i][$j] = $temp;
            }
        }
        return $nilaicossim;
    }

    function normalisasi($table){
        $i=0;
        foreach ($table as &$sample) {
            $sigma = 0.0;
            foreach ($sample as &$value) {
                $sigma += $value ** 2;
            }
            $norm = sqrt($sigma ?: EPSILON);
            $x=0;
            foreach ($sample as &$value) {
                $value /= $norm;
                $table_baru[$i][$x]=($value);
                $x++;
            }
            $i++;
        }
        return $table_baru;
    }
    
    function minmax($table){
        $min = min($table);
        $max = max($table);
        $new_min = 0;
        $new_max = 1;
        foreach ($table as $i => $v) {
            $table_baru[$i] = ($max - $min) != 0 ? ((($new_max - $new_min) * ($v - $min)) / ($max - $min)) + $new_min : (($new_max - $new_min) * ($v - $min)) + $new_min ;
        }
        return $table_baru;
    }

    ?>
