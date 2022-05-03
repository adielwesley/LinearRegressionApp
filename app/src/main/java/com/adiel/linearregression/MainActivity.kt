package com.adiel.linearregression

import android.os.Bundle
import android.util.Log
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import com.adiel.linearregression.databinding.ActivityMainBinding
import com.adiel.linearregression.ml.LinearRegressionMd
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private lateinit var model: LinearRegressionMd

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)

        model = LinearRegressionMd.newInstance(this)
        binding.goButton.setOnClickListener {
            onGoClicked()
        }
    }

    private fun onGoClicked() = runBlocking {
        withContext(Dispatchers.IO) {
            (binding.xTextNumberDecimal as? EditText)?.let { input ->

                // handle input
                val x = input.text.toString().ifEmpty { "0" }.toFloat()
                val xValue = TensorBuffer.createFixedSize(intArrayOf(1, 1), DataType.FLOAT32)
                xValue.loadArray(FloatArray(1) { x })

                // run model
                val outputs = model.process(xValue)

                // handle output
                val yValue = outputs.yValueAsTensorBuffer.getFloatValue(0)
                Log.d("result", "y = $yValue")
                runOnUiThread { binding.yTextView.text = String.format("%.2f", yValue) }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }
}