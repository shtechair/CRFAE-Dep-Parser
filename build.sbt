name := "CRFAE-Dep-Parser"

organization := "edu.shanghaitech.nlp"

version := "1.0"

scalaVersion := "2.12.1"

// project description
description := "CRF autoencoder for unsupervised dependency parsing"

// library dependencies. (orginization name) % (project name) % (version)
libraryDependencies ++= Seq(
  "edu.stanford.nlp" % "stanford-corenlp" % "3.8.0",
  "org.tinylog" % "tinylog" % "1.2",
  "net.sourceforge.argparse4j" % "argparse4j" % "0.7.0"
)

// This project is written in Java, so we excluded the scala-*.jar
assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)



