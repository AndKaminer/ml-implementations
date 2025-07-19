provider "aws" {
  region = var.region
}

resource "aws_s3_bucket" "app_bucket" {
  bucket = "${var.app_name}-bucket-${random_id.suffix.hex}"
  force_destroy = true
}

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_object" "app_zip" {
  bucket = aws_s3_bucket.app_bucket.id
  key    = "app.zip"
  source = var.app_zip_path
  etag   = filemd5(var.app_zip_path)
}

resource "aws_elastic_beanstalk_application" "app" {
  name = var.app_name
}

resource "aws_elastic_beanstalk_application_version" "version" {
  name        = "${var.app_name}-v1"
  application = aws_elastic_beanstalk_application.app.name
  bucket      = aws_s3_bucket.app_bucket.id
  key         = aws_s3_object.app_zip.id
}

resource "aws_elastic_beanstalk_environment" "env" {
  name                = "${var.app_name}-env"
  application         = aws_elastic_beanstalk_application.app.name
  solution_stack_name = "64bit Amazon Linux 2 v3.5.10 running Docker"
  version_label       = aws_elastic_beanstalk_application_version.version.name

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = "t3.micro"
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment"
    name      = "EnvironmentType"
    value     = "SingleInstance"
  }
}

