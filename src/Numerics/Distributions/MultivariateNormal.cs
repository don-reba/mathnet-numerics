// <copyright file="Multinomial.cs" company="Math.NET">
// Math.NET Numerics, part of the Math.NET Project
// http://numerics.mathdotnet.com
// http://github.com/mathnet/mathnet-numerics
//
// Copyright (c) 2009-2013 Math.NET
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// </copyright>

using System;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using MathNet.Numerics.Statistics;

using static System.Math;

namespace MathNet.Numerics.Distributions
{
    /// <summary>
    /// Multivariate Normal distribution. For details about this distribution, see
    /// <a href="https://en.wikipedia.org/wiki/Multivariate_normal_distribution">Wikipedia - Multivariate normal distribution</a>.
    /// Supports degenerate (non-singular) covariance matrices.
    /// </summary>
    [DebuggerDisplay("{Mean.Count}D Normal")]
    public class MultivariateNormal : IDistribution
    {
        System.Random _random;

        readonly Vector<double> _mean;
        readonly Matrix<double> _covariance;
        readonly Matrix<double> _root;

        public MultivariateNormal(Vector<double> mean, Matrix<double> covariance, System.Random randomSource)
        {
            if (!IsValidParameterSet(mean, covariance))
            {
                throw new ArgumentException("Invalid parametrization for the distribution.");
            }

            _random = randomSource;
            _mean = mean.Clone();
            _covariance = covariance.Clone();
            _root = CovarianceRoot(covariance);
        }

        /// <summary>
        /// Initializes a standard multivariate normal distribution with a zero mean and an identity covariance.
        /// </summary>
        public MultivariateNormal(int dimensions)
            : this(dimensions, SystemRandomSource.Default)
        {
        }

        /// <summary>
        /// Initializes a standard multivariate normal distribution with a zero mean and an identity covariance.
        /// </summary>
        public MultivariateNormal(int dimensions, System.Random randomSource)
        {
            if (!IsValidParameterSet(dimensions))
            {
                throw new ArgumentException("Invalid parametrization for the distribution.");
            }

            _random = randomSource;
            _mean = StandardMean(dimensions);
            _covariance = StandardCovariance(dimensions);
            _root = _covariance;

        }

        public MultivariateNormal(Vector<double> mean, Matrix<double> covariance)
            : this(mean, covariance, SystemRandomSource.Default)
        {
        }

        /// <summary>
        /// A string representation of the distribution.
        /// </summary>
        /// <returns>a string representation of the distribution.</returns>
        public override string ToString()
        {
            return $"Normal(μ = {_mean}, Σ = {_covariance})";
        }

        /// <summary>
        /// Tests whether the provided values are valid parameters for a standard multivariate normal distribution.
        /// </summary>
        public static bool IsValidParameterSet(int dimensions)
        {
            return dimensions >= 0;
        }

        /// <summary>
        /// Tests whether the provided values are valid parameters for a multivariate normal distribution.
        /// </summary>
        public static bool IsValidParameterSet(Vector<double> mean, Matrix<double> covariance)
        {
            if (mean == null || covariance == null)
                return false;

            // dimensions must agree
            if (mean.Count != covariance.RowCount || mean.Count != covariance.ColumnCount)
                return false;

            // need no further checks for a zero-dimensional distribution
            if (mean.Count == 0)
                return true;

            // covariance must be symmetric
            if (!covariance.IsSymmetric())
                return false;

            return true;
        }

        public Vector<double> Mean => _mean;

        public Vector<double> Mode => _mean;

        public Matrix<double> Covariance => _covariance;

        /// <summary>
        /// Gets the multivariate normal precision, equal to the inverse of the covariance.
        /// </summary>
        public Matrix<double> Precision => _covariance.Inverse();

        /// <summary>
        /// Computes the probability density of the distribution (PDF) at x, equal to exp(-½(x-μ)ᵀΣ⁻¹(x-μ)) / sqrt(det(Σ)(2π)^k). Defined only if the covariance is positive definite.
        /// </summary>
        public double Density(Vector<double> x)
        {
            return Exp(-0.5 * (x - _mean) * _covariance.Inverse() * (x - _mean))
                / Sqrt(_covariance.Determinant() * Pow(2.0 * PI, _mean.Count));
        }

        /// <summary>
        /// Gets or sets the random number generator which is used to draw random samples.
        /// </summary>
        public System.Random RandomSource
        {
            get => _random;
            set => _random = value ?? SystemRandomSource.Default;
        }

        /// <summary>
        /// Samples a multivariate normal random variable.
        /// </summary>
        /// <returns>the counts for each of the different possible values.</returns>
        public Vector<double> Sample()
        {
            return SampleFromRoot(_random, _mean, _root);
        }

        /// <summary>
        /// Generates a sequence of samples from a multivariate normal variable.
        /// </summary>
        public IEnumerable<Vector<double>> Samples()
        {
            while (true)
            {
                yield return SampleFromRoot(_random, _mean, _root);
            }
        }

        /// <summary>
        /// Computes a square root A of the matrix M, such that AᵀA = M.
        /// </summary>
        private static Matrix<double> CovarianceRoot(Matrix<double> m)
        {
            if (m.ColumnCount == 0 && m.RowCount == 0)
                return m;
            var svd = m.Svd(true);
            return svd.U * svd.W.PointwiseSqrt();
        }

        private static Vector<double> SampleFromRoot(System.Random rnd, Vector<double> mean, Matrix<double> root)
        {
            if (mean.Count == 0)
                return mean;

            // choose x from the standard multivariate normal distribution
            var x = Vector<double>.Build.SameAs(mean);
            for (int i = 0; i != x.Count; ++i)
                x[i] = Normal.Sample(rnd, 0.0, 1.0);

            return mean + root * x;
        }

        /// <summary>
        /// Samples a multivariate normal random variable.
        /// </summary>
        public static Vector<double> Sample(System.Random rnd, Vector<double> mean, Matrix<double> covariance)
        {
            return SampleFromRoot(rnd, mean, CovarianceRoot(covariance));
        }

        /// <summary>
        /// Samples a standard multivariate normal random variable.
        /// </summary>
        /// <remarks>Computationally slightly more efficient than the general case.</remarks>
        public static Vector<double> Sample(System.Random rnd, int dimensions)
        {
            var x = Vector<double>.Build.Dense(dimensions);

            for (int i = 0; i != x.Count; ++i)
                x[i] = Normal.Sample(rnd, 0.0, 1.0);

            return x;
        }

        /// <summary>
        /// Generates a sequence of samples from a multivariate normal variable.
        /// </summary>
        /// <returns>a sequence of counts for each of the different possible values.</returns>
        public static IEnumerable<Vector<double>> Samples(System.Random rnd, Vector<double> mean, Matrix<double> covariance)
        {
            while (true)
            {
                yield return Sample(rnd, mean, covariance);
            }
        }

        /// <summary>
        /// Generates a sequence of samples from a standard multivariate normal variable.
        /// </summary>
        public static IEnumerable<Vector<double>> Samples(System.Random rnd, int dimensions)
        {
            while (true)
            {
                yield return Sample(rnd, dimensions);
            }
        }

        public static Vector<double> StandardMean(int dimensions)
        {
            return Vector<double>.Build.Dense(dimensions);
        }

        public static Matrix<double> StandardCovariance(int dimensions)
        {
            return Matrix<double>.Build.DiagonalIdentity(dimensions);
        }

        /// <summary>
        /// Multivariate normality test using the Henze-Zerkel's test statistic.
        /// N. Henze, B. Zirkler - A class of invariant and consistent tests for multivariate normality - 1990
        /// </summary>
        /// <see href="link">https://cran.r-project.org/package=MVN</see>
        /// <remarks>This is a good default test because of its affine invariance and consistency. This is to say that an affine transformation of the input does not change the test's outcome, and, given enough data, the test will reject any distribution other than the multivariate normal.</remarks>
        /// <returns></returns>
        public static (double Value, double Significance, bool Pass) HenzeZirklerTest(IEnumerable<Vector<double>> samples)
        {
            double n = samples.Count();

            var (m, cov) = samples.MeanCovariance();

            var covInv = ((n - 1) * cov / n).PseudoInverse();
            double p = covInv.Rank();
 
            // smoothing parameter
            double b = Constants.Sqrt1Over2 * Pow((p + p + 1) / 4, 1 / (p + 4)) * Pow(n, 1 / (p + 4));

            // test statistic

            double sum1 = 0;
            foreach (var sampleI in samples)
            foreach (var sampleJ in samples)
            {
                var diff = sampleI - sampleJ;
                double scaledResidual = diff * covInv * diff;
                sum1 += Exp(-0.5 * b * b * scaledResidual);
            }

            double sum2 = 0;
            foreach (var sample in samples)
            {
                var diff = sample - m;
                double scaledResidual = diff * covInv * diff;
                sum2 += Exp(-0.5 * b * b * scaledResidual / (1 + b * b));
            }

            double hz = n * (
                    1 / n / n * sum1 -
                    2 * Pow(1 + b * b, -0.5 * p) * 1 / n * sum2 +
                    Pow(1 + 2 * b * b, -0.5 * p));

            // statistic distribution

            double wb = (1 + b * b) * (1 + 3 * b * b);

            double a = 1 + 2 * b * b;

            double hzMean = 1 - Pow(a, -0.5 * p) * (1 + p * b * b / a + p * (p + 2) * Pow(b, 4) / (2 * a * a));
            double hzVar =
                2 * Pow(1 + 4 * b * b, -0.5 * p) +
                2 * Pow(a, -p) * (1 + 2 * p * Pow(b, 4) / a / a + 3 * p * (p + 2) * Pow(b, 8) / 4 / Pow(a, 4)) -
                4 * Pow(wb, -0.5 * p) * (1 + 3 * p * Pow(b, 4) / (2 * wb) + p * (p + 2) * Pow(b, 8) / (2 * wb * wb));

            // log-Normal distribution

            double logNormalHzMean = Log(Sqrt(Pow(hzMean, 4) / (hzVar + hzMean * hzMean)));
            double logNormalHzStdDev = Sqrt(Log((hzVar + hzMean * hzMean) / hzMean / hzMean));
            double pValue = 1 - LogNormal.CDF(logNormalHzMean, logNormalHzStdDev, hz);

            return (hz, pValue, pValue > 0.05);
        }
    }
}
