// <copyright file="MultinomialTests.cs" company="Math.NET">
// Math.NET Numerics, part of the Math.NET Project
// http://numerics.mathdotnet.com
// http://github.com/mathnet/mathnet-numerics
//
// Copyright (c) 2009-2016 Math.NET
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
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;
using NUnit.Framework;

using SCG = System.Collections.Generic;

namespace MathNet.Numerics.UnitTests.DistributionTests.Multivariate
{
    /// <summary>
    /// Multinomial distribution tests.
    /// </summary>
    [TestFixture, Category("Distributions"), TestOf("MultivariateNormal")]
    public class MultivariateNormalTests
    {
        [TestCase(0)]
        [TestCase(1)]
        [TestCase(2)]
        [TestCase(int.MaxValue)]
        public void ValidStandardDistributionParameters(int dim)
        {
            Assert.IsTrue(MultivariateNormal.IsValidParameterSet(dim));
        }

        [TestCase(-1)]
        public void InalidStandardDistributionParameters(int dim)
        {
            Assert.IsFalse(MultivariateNormal.IsValidParameterSet(dim));
        }

        [Test]
        public void CanCreateStandardDistribution()
        {
            var distribution = new MultivariateNormal(2);
            Assert.AreEqual(Vector<double>.Build.Dense(2), distribution.Mean);
            Assert.AreEqual(Matrix<double>.Build.DiagonalIdentity(2), distribution.Covariance);
        }

        [TestCase(new double[] { }, new double[] { })]
        [TestCase(new double[] { 0 }, new double[] { 0 })]
        [TestCase(new double[] { -1 }, new double[] { 0 })]
        [TestCase(new double[] { 0, 0 }, new double[] { 0, 0, 0, 0 })]
        [TestCase(new double[] { 1, -1 }, new double[] { 1, 0, 0, 1 })]
        public void ValidGeneralDistributionParameters(double[] mean, double[] cov)
        {
            Assert.IsTrue(MultivariateNormal.IsValidParameterSet(
                Vector<double>.Build.DenseOfArray(mean),
                Matrix<double>.Build.DenseOfRowMajor(mean.Length, mean.Length, cov)));
        }

        [TestCase(new double[] { 0, 0 }, new double[] { 0, 1, 0, 0 }, Description = "Non-symmetric.")]
        public void InvalidGeneralDistributionParameters(double[] mean, double[] cov)
        {
            Assert.IsFalse(MultivariateNormal.IsValidParameterSet(
                Vector<double>.Build.DenseOfArray(mean),
                Matrix<double>.Build.DenseOfRowMajor(mean.Length, mean.Length, cov)));
        }

        [TestCase(1, 0, 0)]
        [TestCase(0, 1, 1)]
        [TestCase(1, 1, 2)]
        [TestCase(1, 2, 1)]
        public void InvalidGeneralDistributionParametersWithMismatchedDimensions(int meanDim, int covRows, int covCols)
        {
            Assert.IsFalse(MultivariateNormal.IsValidParameterSet(
                Vector<double>.Build.Dense(meanDim),
                Matrix<double>.Build.Dense(covRows, covCols)));
        }

        [Test]
        public void CanCreateGeneralDistribution()
        {
            var mean = Vector<double>.Build.DenseOfArray(new double[] { 1.0, 2.5 });
            var covariance = Matrix<double>.Build.DenseOfArray(new double[,] { { 2.0, 1.0 }, { 1.0, 3.0 } });
            var distribution = new MultivariateNormal(mean, covariance);
            Assert.AreEqual(mean, distribution.Mean);
            Assert.AreEqual(covariance, distribution.Covariance);
        }

        [TestCase(new double[] { }, new double[] { })]
        [TestCase(new double[] { 2.0 }, new double[] { 0.5 })]
        [TestCase(new double[] { 2.0, 0.0, 0.0, 4.0 }, new double[] { 0.5, 0.0, 0.0, 0.25 })]
        public void ValidatePrecision(double[] cov, double[] precision)
        {
            var dim = (int)Math.Sqrt(cov.Length);
            var distribution = new MultivariateNormal(
                Vector<double>.Build.Dense(dim),
                Matrix<double>.Build.DenseOfRowMajor(dim, dim, cov));
            AssertHelpers.AlmostEqual(
                Matrix<double>.Build.DenseOfRowMajor(dim, dim, precision),
                distribution.Precision, 4);
        }

        // let dnorm() be the standard normal probability density function
        [TestCase(new double[] { 0 }, new double[] { 1 }, new double[] { 0 }, 0.3989422804014327)] // dnorm(0)
        [TestCase(new double[] { 0 }, new double[] { 1 }, new double[] { 1 }, 0.24197072451914337)] // dnorm(1)
        [TestCase(new double[] { 0 }, new double[] { 1 }, new double[] { 2 }, 0.05399096651318806)] // dnorm(2)
        [TestCase(new double[] { -1, -1 }, new double[] { 2, -1, -1, 2 }, new double[] { 0, 0 }, 0.033803760991572902)] // dnorm(√2) / √6π
        public void ValidateDensity(double[] mean, double[] cov, double[] x, double density)
        {
            var distribution = new MultivariateNormal(
                Vector<double>.Build.DenseOfArray(mean),
                Matrix<double>.Build.DenseOfRowMajor(mean.Length, mean.Length, cov));
            AssertHelpers.AlmostEqual(density, distribution.Density(Vector<double>.Build.DenseOfArray(x)));
        }

        [TestCase(0)]
        [TestCase(1)]
        [TestCase(2)]
        public void StandardAndGeneralSamplesAgree(int dim)
        {
            var standard = new MultivariateNormal(dim, new System.Random(0));
            var general  = new MultivariateNormal(MultivariateNormal.StandardMean(dim), MultivariateNormal.StandardCovariance(dim), new System.Random(0));
            for (int i = 0; i != 5; ++i)
            {
                AssertHelpers.AlmostEqual(
                    standard.Sample().PointwiseAbs(),
                    general.Sample().PointwiseAbs(), 7);
            }
        }

        [TestCase(0)]
        [TestCase(1)]
        [TestCase(2)]
        public void StandardAndGeneralSamplesAgreeStatic(int dim)
        {
            var standardRandom = new System.Random(0);
            var generalRandom = new System.Random(0);
            var mean = MultivariateNormal.StandardMean(dim);
            var covariance = MultivariateNormal.StandardCovariance(dim);
            for (int i = 0; i != 5; ++i)
            {
                AssertHelpers.AlmostEqual(
                    MultivariateNormal.Sample(standardRandom, dim).PointwiseAbs(),
                    MultivariateNormal.Sample(generalRandom, mean, covariance).PointwiseAbs(), 7);
            }
        }

        [TestCase(1, Category = "LongRunning")]
        [TestCase(2, Category = "LongRunning")]
        [TestCase(3, Category = "LongRunning")]
        public void ValidateSamples(int dim)
        {
            var samples = MultivariateNormal.Samples(new System.Random(0), dim).Take(500);
            var (Value, Significance, Pass) = MultivariateNormal.HenzeZirklerTest(samples);
            Assert.True(Pass, $"Value = {Value}; Significance = {Significance}");
        }

        [TestCase(1)]
        [TestCase(2)]
        public void SamplesAreDeterministic(int dim)
        {
            Assert.AreEqual(
                MultivariateNormal.Samples(new System.Random(0), dim).Take(100),
                MultivariateNormal.Samples(new System.Random(0), dim).Take(100));
        }

        [Test]
        // Test against the R MVN implementation.
        public void ValidateHenzeZerkelTest()
        {
            const int equalityDecimals = 7;

            // the extra zero column tests singular covariance support
            var irisSetosa = Matrix<double>.Build.DenseOfArray(new double[,] {
                {5.1, 3.5, 1.4, .2, 0}, {4.9, 3.0, 1.4, .2, 0}, {4.7, 3.2, 1.3, .2, 0}, {4.6, 3.1, 1.5, .2, 0}, {5.0, 3.6, 1.4, .2, 0},
                {5.4, 3.9, 1.7, .4, 0}, {4.6, 3.4, 1.4, .3, 0}, {5.0, 3.4, 1.5, .2, 0}, {4.4, 2.9, 1.4, .2, 0}, {4.9, 3.1, 1.5, .1, 0},
                {5.4, 3.7, 1.5, .2, 0}, {4.8, 3.4, 1.6, .2, 0}, {4.8, 3.0, 1.4, .1, 0}, {4.3, 3.0, 1.1, .1, 0}, {5.8, 4.0, 1.2, .2, 0},
                {5.7, 4.4, 1.5, .4, 0}, {5.4, 3.9, 1.3, .4, 0}, {5.1, 3.5, 1.4, .3, 0}, {5.7, 3.8, 1.7, .3, 0}, {5.1, 3.8, 1.5, .3, 0},
                {5.4, 3.4, 1.7, .2, 0}, {5.1, 3.7, 1.5, .4, 0}, {4.6, 3.6, 1.0, .2, 0}, {5.1, 3.3, 1.7, .5, 0}, {4.8, 3.4, 1.9, .2, 0},
                {5.0, 3.0, 1.6, .2, 0}, {5.0, 3.4, 1.6, .4, 0}, {5.2, 3.5, 1.5, .2, 0}, {5.2, 3.4, 1.4, .2, 0}, {4.7, 3.2, 1.6, .2, 0},
                {4.8, 3.1, 1.6, .2, 0}, {5.4, 3.4, 1.5, .4, 0}, {5.2, 4.1, 1.5, .1, 0}, {5.5, 4.2, 1.4, .2, 0}, {4.9, 3.1, 1.5, .2, 0},
                {5.0, 3.2, 1.2, .2, 0}, {5.5, 3.5, 1.3, .2, 0}, {4.9, 3.6, 1.4, .1, 0}, {4.4, 3.0, 1.3, .2, 0}, {5.1, 3.4, 1.5, .2, 0},
                {5.0, 3.5, 1.3, .3, 0}, {4.5, 2.3, 1.3, .3, 0}, {4.4, 3.2, 1.3, .2, 0}, {5.0, 3.5, 1.6, .6, 0}, {5.1, 3.8, 1.9, .4, 0},
                {4.8, 3.0, 1.4, .3, 0}, {5.1, 3.8, 1.6, .2, 0}, {4.6, 3.2, 1.4, .2, 0}, {5.3, 3.7, 1.5, .2, 0}, {5.0, 3.3, 1.4, .2, 0} }).EnumerateRows();

            var (Value40, Significance40, Pass40) = MultivariateNormal.HenzeZirklerTest(irisSetosa.Take(40));

            // compare to: mvn(iris[1:40, 1:4])
            AssertHelpers.AlmostEqual(0.8562336, Value40, equalityDecimals);
            AssertHelpers.AlmostEqual(0.1503585, Significance40, equalityDecimals);
            Assert.That(Pass40, Is.True);

            var (Value50, Significance50, Pass50) = MultivariateNormal.HenzeZirklerTest(irisSetosa);

            // compare to: mvn(iris[1:50, 1:4])
            AssertHelpers.AlmostEqual(0.9488453, Value50, equalityDecimals);
            AssertHelpers.AlmostEqual(0.0499536, Significance50, equalityDecimals);
            Assert.That(Pass50, Is.False);
        }
    }
}
